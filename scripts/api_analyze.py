import sys, os, json, glob
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B, OUT = sys.argv[1], sys.argv[2]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


frames = []
for f in sorted(glob.glob(os.path.join(B, "api_results", "*__results.jsonl"))):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    d = pd.DataFrame(rows).drop_duplicates("uid")
    frames.append(d)
    print("%s : %d rows" % (os.path.basename(f), len(d)))
api = pd.concat(frames, ignore_index=True)

print("\n" + "=" * 104)
print("API MODEL RESULTS")
print("=" * 104)
res = []
for (mdl, ds, st), g in api.groupby(["model", "dataset", "strategy"]):
    res.append(dict(model=mdl, dataset=ds, strategy=st, n=len(g),
                    acc=round(100 * (g.pred == g.gold).mean(), 2),
                    unparsed=int((g.pred < 0).sum()),
                    p_tok=round(g.prompt_tokens.mean(), 1),
                    o_tok=round(g.completion_tokens.mean(), 2)))
R = pd.DataFrame(res).sort_values(["model", "dataset", "strategy"])
print(R.to_string(index=False))

print("\n" + "=" * 104)
print("PIVOT EFFECT, with exact McNemar on paired items")
print("=" * 104)
out = []
for (mdl, ds), g in api.groupby(["model", "dataset"]):
    base = g[g.strategy == "ar_direct"][["id", "gold", "pred"]].rename(columns={"pred": "pa"})
    if base.empty:
        continue
    b_acc = 100 * (base.pa == base.gold).mean()
    b_tok = g[g.strategy == "ar_direct"].prompt_tokens.mean()
    for st in ("ar_mt_pivot", "en_human"):
        alt = g[g.strategy == st][["id", "pred"]].rename(columns={"pred": "pp"})
        if alt.empty:
            continue
        m = base.merge(alt, on="id")
        oa, op = m.pa == m.gold, m.pp == m.gold
        lost, gain = int((oa & ~op).sum()), int((~oa & op).sum())
        a_acc = 100 * op.mean()
        a_tok = g[g.strategy == st].prompt_tokens.mean()
        out.append(dict(model=mdl, dataset=ds, strategy=st, n=len(m),
                        arabic=round(b_acc, 2), alt=round(a_acc, 2),
                        delta=round(a_acc - b_acc, 2), lost=lost, gained=gain,
                        p="%.2g" % mcnemar(lost, gain),
                        tok_change_pct=round(100 * (a_tok - b_tok) / b_tok, 1)))
P = pd.DataFrame(out)
print(P.to_string(index=False))
P.to_csv(OUT, index=False, encoding="utf-8")

print("\n" + "=" * 104)
print("SIDE BY SIDE WITH THE OPEN MODELS (ArabicMMLU, Arabic -> 600M pivot)")
print("=" * 104)
ref = [("Qwen2.5-7B", 63.07, 47.00), ("Llama-3.1-8B", 54.60, 46.00),
       ("Qwen2.5-14B", 66.33, 46.07)]
for name, ar, pv in ref:
    print("   %-13s %6.2f -> %6.2f   %+6.2f" % (name, ar, pv, pv - ar))
for r in P[(P.dataset == "ammlu") & (P.strategy == "ar_mt_pivot")].itertuples():
    print("   %-13s %6.2f -> %6.2f   %+6.2f   <-- API" % (r.model, r.arabic, r.alt, r.delta))

print("\nwrote", OUT)
