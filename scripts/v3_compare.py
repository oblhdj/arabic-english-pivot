"""Does the 3.3B translator lift the pivot ceiling? Compares, per model and dataset:
Arabic direct -> 600M pivot -> 3B pivot -> human English."""
import sys, os, zipfile
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B = sys.argv[1]
for tag, z in (("rv2", r"C:\Users\Mega-PC\Downloads\results_v2.zip"),
               ("v1full", r"C:\Users\Mega-PC\Downloads\results (1).zip"),
               ("v1rerun", r"C:\Users\Mega-PC\Downloads\rerun_results.zip")):
    d = os.path.join(B, tag)
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(d)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rd(*p):
    f = os.path.join(B, *p)
    return pd.read_parquet(f) if os.path.exists(f) else None


V1 = {"Qwen2.5-7B": "Qwen2.5-7B-Instruct", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct"}


def cond(model, ds, strat):
    if model == "Qwen2.5-14B":
        if strat.startswith("pivot_3b"):
            return rd("rv3", f"Qwen2.5-14B__{ds}__{strat}.parquet")
        return rd("rv2", f"Qwen2.5-14B__{ds}__{strat}.parquet")
    v1 = V1[model]
    if strat == "ar_direct":
        d = rd("v1rerun", f"{v1}__{ds}__ar_direct__direct.parquet")   # rerun supersedes
        return d if d is not None else rd("v1full", f"{v1}__{ds}__ar_direct__direct.parquet")
    if strat == "en_human":
        return rd("v1full", f"{v1}__{ds}__en_human__direct.parquet")
    if strat.startswith("pivot_600m"):
        return rd("rv2", f"{model}__{ds}__{strat}.parquet")
    return rd("rv3", f"{model}__{ds}__{strat}.parquet")


STRATS = ["ar_direct", "pivot_600m_seg", "pivot_3b_seg",
          "pivot_600m_ctx", "pivot_3b_ctx", "en_human"]
rows = []
for model in ("Qwen2.5-7B", "Llama-3.1-8B", "Qwen2.5-14B"):
    for ds in ("belebele", "ammlu"):
        for st in STRATS:
            d = cond(model, ds, st)
            if d is None:
                continue
            rows.append(dict(model=model, dataset=ds, strategy=st, n=len(d),
                             acc=round(100 * (d.pred == d.gold).mean(), 2),
                             p_tok=round(d.prompt_tokens.mean(), 1)))
R = pd.DataFrame(rows)
print("=" * 94)
print("ACCURACY BY STRATEGY  (600M -> 3B is the question)")
print("=" * 94)
print(R.pivot_table(index=["dataset", "model"], columns="strategy", values="acc")
      .reindex(columns=STRATS).round(2).to_string())
print("\nmean prompt tokens:")
print(R.pivot_table(index=["dataset", "model"], columns="strategy", values="p_tok")
      .reindex(columns=STRATS).round(1).to_string())

print("\n" + "=" * 94)
print("DID 3B CLOSE THE GAP?  paired on item id, exact McNemar")
print("=" * 94)
out = []
for model in ("Qwen2.5-7B", "Llama-3.1-8B", "Qwen2.5-14B"):
    for ds in ("belebele", "ammlu"):
        for variant in ("seg", "ctx"):
            a = cond(model, ds, f"pivot_600m_{variant}")
            b = cond(model, ds, f"pivot_3b_{variant}")
            if a is None or b is None:
                continue
            m = a[["id", "gold", "pred"]].rename(columns={"pred": "p6"}).merge(
                b[["id", "pred"]].rename(columns={"pred": "p3"}), on="id")
            o6, o3 = m.p6 == m.gold, m.p3 == m.gold
            lost, gain = int((o6 & ~o3).sum()), int((~o6 & o3).sum())
            ar = cond(model, ds, "ar_direct")
            hu = cond(model, ds, "en_human")
            ar_acc = 100 * (ar.pred == ar.gold).mean() if ar is not None else float("nan")
            a6, a3 = 100 * o6.mean(), 100 * o3.mean()
            ceiling = 100 * (hu.pred == hu.gold).mean() if hu is not None else ar_acc
            closed = 100 * (a3 - a6) / (ceiling - a6) if ceiling - a6 > 1e-9 else float("nan")
            out.append(dict(model=model, dataset=ds, variant=variant, n=len(m),
                            m600=round(a6, 2), m3b=round(a3, 2), delta=round(a3 - a6, 2),
                            lost=lost, gained=gain, p="%.2g" % mcnemar(lost, gain),
                            arabic=round(ar_acc, 2), ceiling=round(ceiling, 2),
                            pct_gap_closed=round(closed, 1)))
G = pd.DataFrame(out)
print(G.to_string(index=False))
G.to_csv(sys.argv[2], index=False, encoding="utf-8")
print("\nceiling = human English where it exists (belebele), else Arabic direct (ammlu)")
print("pct_gap_closed = (3B - 600M) / (ceiling - 600M) * 100")
print("\nwrote", sys.argv[2])
