"""Export the exact v1 prompt set so an API run reproduces the Kaggle conditions.

Prompts are taken verbatim from the v1 result parquets. They contain no chat
template, so each API client applies its own. Cross-model identity was verified:
both v1 models were sent byte-identical prompts for every condition.
"""
import sys, os, json, hashlib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

SRC = "Qwen2.5-7B-Instruct"          # prompts identical across models; pick either
CORE = [("belebele", "ar_direct", "direct", "v1full", 900),
        ("belebele", "ar_mt_pivot", "direct", "v1full", 900),
        ("belebele", "en_human", "direct", "v1full", 900),
        ("ammlu", "ar_direct", "direct", "v1rerun", 1500),
        ("ammlu", "ar_mt_pivot", "direct", "v1full", 1500)]
COT = [("belebele", "ar_direct", "cot", "v1full", 300),
       ("belebele", "ar_mt_pivot", "cot", "v1full", 300),
       ("belebele", "en_human", "cot", "v1full", 300)]


def dump(spec, fname, max_tokens):
    path = os.path.join(OUT, fname)
    man, n = [], 0
    with open(path, "w", encoding="utf-8") as fh:
        for ds, st, mode, tag, expect in spec:
            p = os.path.join(B, tag, f"{SRC}__{ds}__{st}__{mode}.parquet")
            d = pd.read_parquet(p).sort_values("id")
            assert len(d) == expect, f"{fname}: {ds}/{st} has {len(d)}, expected {expect}"
            for r in d.itertuples():
                fh.write(json.dumps({
                    "uid": f"{ds}|{st}|{mode}|{r.id}",
                    "dataset": ds, "strategy": st, "mode": mode,
                    "id": int(r.id), "gold": int(r.gold),
                    "prompt": str(r.prompt),
                    "max_new_tokens": max_tokens,
                }, ensure_ascii=False) + "\n")
            blob = "".join(d.prompt.astype(str))
            man.append({"dataset": ds, "strategy": st, "mode": mode, "n": len(d),
                        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                        "mean_prompt_chars": round(d.prompt.astype(str).str.len().mean(), 1)})
            n += len(d)
    return path, man, n


core_path, core_man, core_n = dump(CORE, "prompts_direct.jsonl", 16)
cot_path, cot_man, cot_n = dump(COT, "prompts_cot.jsonl", 512)

manifest = {
    "source": "v1 Kaggle run (results (1).zip + rerun_results.zip)",
    "note": "Prompts carry no chat template; apply the API client's own. "
            "Prompts are identical across the two v1 models, verified byte-for-byte. "
            "ammlu/ar_direct comes from rerun_results.zip, which supersedes the "
            "OOM-truncated run in results (1).zip.",
    "answer_parsing": "v1 parsed with regex; unparsable counts as wrong.",
    "files": {"prompts_direct.jsonl": {"total": core_n, "conditions": core_man},
              "prompts_cot.jsonl": {"total": cot_n, "conditions": cot_man}},
}
mpath = os.path.join(OUT, "manifest.json")
with open(mpath, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)

print("prompts_direct.jsonl : %d prompts" % core_n)
for c in core_man:
    print("   %-9s %-12s n=%-5d sha256=%s" % (c["dataset"], c["strategy"], c["n"], c["sha256"][:16]))
print("prompts_cot.jsonl    : %d prompts" % cot_n)
for c in cot_man:
    print("   %-9s %-12s n=%-5d sha256=%s" % (c["dataset"], c["strategy"], c["n"], c["sha256"][:16]))
print("\nmanifest:", mpath)
for f in sorted(os.listdir(OUT)):
    print("   %-22s %8.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))

# round-trip check
back = [json.loads(l) for l in open(core_path, encoding="utf-8")]
print("\nround-trip: %d lines, %d unique uids, golds in %s"
      % (len(back), len({b["uid"] for b in back}), sorted({b["gold"] for b in back})))
print("sample prompt (pivot):", repr(next(b["prompt"] for b in back
                                          if b["strategy"] == "ar_mt_pivot")[:160]))
