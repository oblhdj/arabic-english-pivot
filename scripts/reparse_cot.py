"""Re-parse reasoning-mode answers from stored output text.

The runner's parse_answer() takes the FIRST digit 1-5, which is correct for
direct mode (the reply is literally "3") but wrong for chain-of-thought, where
the first digit is some number inside the reasoning. Reasoning replies end with
an explicit marker instead:
    English prompts -> "Answer: N"
    Arabic prompts  -> the Arabic word for answer, then N
Both verified on 100% of the relevant outputs. No API calls are made here.
"""
import sys, os, json, re, glob
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SRC, OUT = sys.argv[1], sys.argv[2]

MARK = re.compile(r"(?:answer|الإجابة|الجواب)"
                  r"\s*[:\-]?\s*\**\s*([1-5])", re.I)
TRAIL = re.compile(r"([1-5])\s*\**\s*$")


def reparse(text):
    if not text:
        return -1, "empty"
    m = MARK.findall(text)
    if m:
        return int(m[-1]), "marker"          # LAST marker, not the first
    t = TRAIL.search(text.strip())
    if t:
        return int(t.group(1)), "trailing"
    return -1, "unparsed"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


for path in sorted(glob.glob(os.path.join(SRC, "*__results.jsonl"))):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    how, changed = {}, 0
    for x in rows:
        new, mode = reparse(x.get("output"))
        how[mode] = how.get(mode, 0) + 1
        if new != x["pred"]:
            changed += 1
        x["pred_old"], x["pred"] = x["pred"], new
    print("%s: %d rows, %d predictions corrected, methods=%s"
          % (os.path.basename(path), len(rows), changed, how))

    fixed = path.replace(".jsonl", "__reparsed.jsonl")
    with open(fixed, "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    d = pd.DataFrame(rows)
    print("\n%-14s %5s %8s %8s %9s %9s" % ("strategy", "n", "acc_old", "acc_new", "in_tok", "out_tok"))
    for st in ("ar_direct", "ar_mt_pivot", "en_human"):
        g = d[d.strategy == st]
        if g.empty:
            continue
        print("%-14s %5d %7.2f%% %7.2f%% %9.1f %9.1f"
              % (st, len(g), 100 * (g.pred_old == g.gold).mean(), 100 * (g.pred == g.gold).mean(),
                 g.prompt_tokens.mean(), g.completion_tokens.mean()))

    base = d[d.strategy == "ar_direct"][["id", "gold", "pred"]].rename(columns={"pred": "pa"})
    res = []
    for st in ("ar_mt_pivot", "en_human"):
        alt = d[d.strategy == st][["id", "pred"]].rename(columns={"pred": "pp"})
        if alt.empty:
            continue
        m = base.merge(alt, on="id")
        oa, op = m.pa == m.gold, m.pp == m.gold
        lost, gain = int((oa & ~op).sum()), int((~oa & op).sum())
        bt = d[d.strategy == "ar_direct"]
        at = d[d.strategy == st]
        w_b = bt.prompt_tokens.mean() + 4 * bt.completion_tokens.mean()
        w_a = at.prompt_tokens.mean() + 4 * at.completion_tokens.mean()
        res.append(dict(strategy=st, n=len(m), arabic=round(100 * oa.mean(), 2),
                        alt=round(100 * op.mean(), 2), delta=round(100 * (op.mean() - oa.mean()), 2),
                        lost=lost, gained=gain, p="%.2g" % mcnemar(lost, gain),
                        weighted_tok_change=round(100 * (w_a - w_b) / w_b, 1)))
    R = pd.DataFrame(res)
    print()
    print(R.to_string(index=False))
    R.to_csv(OUT, index=False, encoding="utf-8")
    print("\nwrote", OUT)
