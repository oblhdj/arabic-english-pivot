import sys, os, glob
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]

rows = []
for p in sorted(glob.glob(os.path.join(D, "*__*__*.parquet"))):
    df = pd.read_parquet(p)
    base = os.path.basename(p)[:-len(".parquet")]
    model, dataset, strategy = base.split("__")
    bad = int((df.pred < 0).sum())
    rows.append(dict(
        model=model, dataset=dataset, strategy=strategy, n=len(df),
        acc=round(100 * (df.pred == df.gold).mean(), 2),
        acc_parsed=round(100 * (df.loc[df.pred >= 0, "pred"] == df.loc[df.pred >= 0, "gold"]).mean(), 2) if bad else None,
        unparsed=bad,
        p_tok=round(df.prompt_tokens.mean(), 1),
        p_tok_sum=int(df.prompt_tokens.sum()),
        g_tok=round(df.gen_tokens.mean(), 1),
    ))

R = pd.DataFrame(rows).sort_values(["dataset", "model", "strategy"])
print("=" * 100)
print("PER-RUN RESULTS  (acc = accuracy %, p_tok = mean prompt tokens)")
print("=" * 100)
print(R.to_string(index=False))

# completeness
models = sorted(R.model.unique())
datasets = sorted(R.dataset.unique())
strats = sorted(R.strategy.unique())
have = set(zip(R.model, R.dataset, R.strategy))
missing = [(m, d, s) for m in models for d in datasets for s in strats if (m, d, s) not in have]
print("\n" + "=" * 100)
print("GRID: %d models x %d datasets x %d strategies = %d cells; have %d"
      % (len(models), len(datasets), len(strats), len(models) * len(datasets) * len(strats), len(have)))
print("strategies:", strats)
print("\nabsent cells (some never applied, e.g. en_human only exists for belebele):")
for m, d, s in missing:
    print("   -", m, d, s)

# pivot vs baseline, where a baseline exists in THIS zip
print("\n" + "=" * 100)
print("PIVOT vs ar_direct  (same model+dataset, within this zip only)")
print("=" * 100)
for (m, d), g in R.groupby(["model", "dataset"]):
    b = g[g.strategy == "ar_direct"]
    if b.empty:
        continue
    b = b.iloc[0]
    print("\n%s / %s   baseline ar_direct: acc %.2f, %.1f prompt tokens" % (m, d, b.acc, b.p_tok))
    for _, r in g[g.strategy != "ar_direct"].iterrows():
        print("   %-18s acc %6.2f  (%+6.2f)   p_tok %7.1f  (%+6.1f%%)"
              % (r.strategy, r.acc, r.acc - b.acc, r.p_tok, 100 * (r.p_tok - b.p_tok) / b.p_tok))

# strategy means across models (600m pivots only)
print("\n" + "=" * 100)
print("MEAN ACROSS MODELS, per dataset+strategy")
print("=" * 100)
print(R.groupby(["dataset", "strategy"]).agg(
    runs=("acc", "size"), mean_acc=("acc", "mean"), mean_p_tok=("p_tok", "mean")
).round(2).to_string())

# translation caches
print("\n" + "=" * 100)
print("TRANSLATION CACHES")
print("=" * 100)
for name in ("belebele_translations_v2.parquet", "ammlu_translations_v2.parquet"):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        print(name, "MISSING"); continue
    t = pd.read_parquet(p)
    print("\n==", name, t.shape)
    print("columns:", list(t.columns))
    nulls = t.isna().sum()
    print("null counts:\n" + nulls[nulls > 0].to_string() if nulls.any() else "no nulls")
    for c in t.columns:
        if t[c].dtype == object and t[c].notna().any():
            print("   %-14s mean chars %7.1f" % (c, t[c].dropna().astype(str).str.len().mean()))
