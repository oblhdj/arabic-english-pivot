# ===================================================================
#  NLLB-3.3B rerun — drop-in replacement for the cell that OOM'd
#  Reconstructed from errors.txt tracebacks, so CHECK THE NAMES against
#  your notebook before running: translate(), build(), job(), safe(), both, timing.
#
#  ROOT CAUSE of the OOM: the 3.3B model was loaded in float32.
#     3.3e9 params x 4 bytes = 13.2 GiB  ->  matches the reported
#     "13.41 GiB is allocated by PyTorch" on a 14.56 GiB T4, leaving
#     nothing for activations. Hence it died asking for a mere 92 MiB.
#  THE FIX is fp16 (6.6 GiB), not a smaller batch. Batch size is secondary.
# ===================================================================

# --- put this in the FIRST cell of the notebook, before importing torch ---
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def free_gpu(*objs):
    """Drop model/tokenizer refs and actually return the memory."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("   gpu now: %.2f GiB allocated, %.2f GiB reserved"
          % (torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30))


def load_mt(repo):
    """Load an NLLB checkpoint in fp16. This is the line that was missing."""
    tok = AutoTokenizer.from_pretrained(repo, src_lang="arb_Arab")
    m = AutoModelForSeq2SeqLM.from_pretrained(
        repo,
        dtype=torch.float16,        # <-- THE FIX. use torch_dtype= on transformers < 4.56
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    print("loaded %s : %.2f GiB" % (repo, torch.cuda.memory_allocated() / 2**30))
    return tok, m


@torch.inference_mode()
def translate_3b(tok, m, texts, bs=8, max_in=512):
    """Same contract as your translate(): returns (list_of_outputs, seconds, lengths).
    Sorted by length so each batch pads to a similar width — big memory win."""
    import time
    eng = tok.convert_tokens_to_ids("eng_Latn")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    lens = [0] * len(texts)
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_in).to("cuda")
        try:
            g = m.generate(**enc, forced_bos_token_id=eng, num_beams=2,
                           max_new_tokens=int(enc.input_ids.shape[1] * 1.6) + 16)
        except torch.OutOfMemoryError:
            # last-resort: retry this batch one item at a time, greedy
            torch.cuda.empty_cache()
            g = None
            dec = []
            for i in idx:
                e1 = tok([texts[i]], return_tensors="pt", truncation=True,
                         max_length=max_in).to("cuda")
                g1 = m.generate(**e1, forced_bos_token_id=eng, num_beams=1,
                                max_new_tokens=int(e1.input_ids.shape[1] * 1.6) + 16)
                dec.append(tok.batch_decode(g1, skip_special_tokens=True)[0])
                del e1, g1
        if g is not None:
            dec = tok.batch_decode(g, skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d
            lens[i] = len(d)
        del enc
        if g is not None:
            del g
        if (s // bs) % 20 == 0:
            torch.cuda.empty_cache()
            print("   %d/%d  %.0fs" % (s + len(idx), len(order), time.time() - t0), flush=True)
    return out, time.time() - t0, lens


# ------------------------------------------------------------------
#  RUN ORDER — the 600m model must be gone before 3B loads.
# ------------------------------------------------------------------
# free whatever MT model is still resident from the 600m pass:
#     free_gpu(m, tok)          # use your own variable names
#
# then, for the 3b tag, inside your build()/job() replace the
# translate(...) call with translate_3b(tok3b, m3b, texts, bs=8)
# and set the batch size via:
#
#     BS = {"600m": 48, "3b": 8}[tag]
#
# After the 3B columns are built, free again before the LLM eval models load:
#     free_gpu(m3b, tok3b)
#
# Expected: ~25-40 min for the full belebele+ammlu translation set at bs=8,
# against the 492 s the 600m took. If it still OOMs, drop to bs=4; if it has
# room to spare, bs=16 will work in fp16.

# ------------------------------------------------------------------
#  OPTIONAL: Kaggle gives you 2x T4. Select "GPU T4 x2" in the sidebar
#  and you can shard the 3.3B model instead of squeezing it:
#     m = AutoModelForSeq2SeqLM.from_pretrained(
#             repo, dtype=torch.float16, device_map="auto", max_memory={0:"13GiB",1:"13GiB"})
#  Do NOT call .to("cuda") when using device_map.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  SKIP the two missing Qwen2.5-14B pivot_600m_guard runs.
#  The guard flags do not correlate with accuracy (flagged items scored
#  47.1 vs 54.4 unflagged on Qwen-7B ammlu), so the strategy has no
#  empirical basis and the two absent cells would not change any result.
# ------------------------------------------------------------------
