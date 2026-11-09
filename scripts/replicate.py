import sys, os
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B = sys.argv[1]
OUT = sys.argv[2]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---- subject labels, keyed on ArabicMMLU's own question ID (1..14562).
# v1 results and v2 results cover the SAME 1500 questions: identical id sets,
# golds agree on 100% of rows. So one label table serves all three models.
c1 = pd.read_parquet(os.path.join(B, "v1full", "ammlu_with_mt.parquet"))
lab1 = c1[["ID", "Subject", "Group"]].rename(
    columns={"ID": "id", "Subject": "subject", "Group": "group"})
c2 = pd.read_parquet(os.path.join(B, "rv2", "ammlu_translations_v2.parquet"))
lab2 = c2[["id", "subject", "group"]]

chk = lab1.merge(lab2, on="id", suffixes=("_1", "_2"))
print("label tables joined on %d ids; subject strings agree on %.1f%%"
      % (len(chk), 100 * (chk.subject_1 == chk.subject_2).mean()))
lab1 = lab2          # use the single verified label table everywhere

RUNS = [
    ("Qwen2.5-7B", os.path.join(B, "v1rerun", "Qwen2.5-7B-Instruct__ammlu__ar_direct__direct.parquet"),
     os.path.join(B, "v1full", "Qwen2.5-7B-Instruct__ammlu__ar_mt_pivot__direct.parquet"), lab1),
    ("Llama-3.1-8B", os.path.join(B, "v1rerun", "Meta-Llama-3.1-8B-Instruct__ammlu__ar_direct__direct.parquet"),
     os.path.join(B, "v1full", "Meta-Llama-3.1-8B-Instruct__ammlu__ar_mt_pivot__direct.parquet"), lab1),
    ("Qwen2.5-14B", os.path.join(B, "rv2", "Qwen2.5-14B__ammlu__ar_direct.parquet"),
     os.path.join(B, "rv2", "Qwen2.5-14B__ammlu__pivot_600m_seg.parquet"), lab2),
]

per_subject, per_group, headline = {}, {}, []
for name, fa, fp, lab in RUNS:
    a = pd.read_parquet(fa)[["id", "gold", "pred"]].rename(columns={"pred": "pa"})
    p = pd.read_parquet(fp)[["id", "pred"]].rename(columns={"pred": "pp"})
    m = a.merge(p, on="id").merge(lab, on="id")
    m["oa"] = m.pa == m.gold
    m["op"] = m.pp == m.gold
    lost = int((m.oa & ~m.op).sum()); gained = int((~m.oa & m.op).sum())
    headline.append(dict(model=name, n=len(m), ar=round(100 * m.oa.mean(), 2),
                         pivot=round(100 * m.op.mean(), 2),
                         delta=round(100 * (m.op.mean() - m.oa.mean()), 2),
                         lost=lost, gained=gained, p=mcnemar(lost, gained)))
    per_subject[name] = m.groupby("subject").apply(
        lambda g: round(100 * (g.op.mean() - g.oa.mean()), 1), include_groups=False)
    per_group[name] = m.groupby("group").apply(
        lambda g: round(100 * (g.op.mean() - g.oa.mean()), 1), include_groups=False)

print("\n" + "=" * 88)
print("HEADLINE: ArabicMMLU, Arabic-direct vs 600M pivot (sentence-level)")
print("=" * 88)
H = pd.DataFrame(headline)
H["p"] = H.p.map(lambda v: "%.2g" % v)
print(H.to_string(index=False))

print("\n" + "=" * 88)
print("REPLICATION BY GROUP (accuracy change in points)")
print("=" * 88)
G = pd.DataFrame(per_group)
print(G.to_string())

print("\n" + "=" * 88)
print("REPLICATION BY SUBJECT (n>=20 in all three)")
print("=" * 88)
S = pd.DataFrame(per_subject).dropna()
counts = pd.DataFrame(per_subject).join(
    lab2.groupby("subject").size().rename("n"))["n"]
S = S.join(counts)
S = S[S.n >= 20].sort_values("Qwen2.5-14B")
print(S.to_string())

print("\nCross-model agreement on which subjects are hurt most (Spearman):")
print(S[["Qwen2.5-7B", "Llama-3.1-8B", "Qwen2.5-14B"]].corr(method="spearman").round(3).to_string())

S.to_csv(OUT, encoding="utf-8")
print("\nwrote", OUT)
