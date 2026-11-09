import sys, os, glob, zipfile, json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1]
SRC = {
    "v1full": r"C:\Users\Mega-PC\Downloads\results (1).zip",
    "v1pilot": r"C:\Users\Mega-PC\Downloads\results.zip",
    "v1rerun": r"C:\Users\Mega-PC\Downloads\rerun_results.zip",
}
for tag, z in SRC.items():
    d = os.path.join(BASE, tag)
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(d)
    print(tag, "->", len(glob.glob(os.path.join(d, "*.parquet"))), "parquets")

print("\n" + "=" * 90)
print("ROW COUNTS: pilot vs full vs rerun  (identifies which is authoritative)")
print("=" * 90)
name = "Qwen2.5-7B-Instruct__ammlu__ar_direct__direct.parquet"
for tag in ("v1pilot", "v1full", "v1rerun"):
    p = os.path.join(BASE, tag, name)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        acc = 100 * (df.pred == df.gold).mean() if "pred" in df else float("nan")
        print("  %-8s %-10s rows=%5d  acc=%.2f" % (tag, "ammlu/ar", len(df), acc))
    else:
        print("  %-8s absent" % tag)

print("\nschema of a v1 result file:")
df = pd.read_parquet(os.path.join(BASE, "v1full",
                                  "Qwen2.5-7B-Instruct__belebele__ar_direct__direct.parquet"))
print(df.dtypes.to_string())

print("\nschema of ammlu_with_mt.parquet (v1 translation cache):")
t = pd.read_parquet(os.path.join(BASE, "v1full", "ammlu_with_mt.parquet"))
print("shape", t.shape)
print("columns:", list(t.columns))

print("\nall v1 runs (v1full + v1rerun; rerun supersedes):")
paths = {}
for tag in ("v1full", "v1rerun"):
    for p in sorted(glob.glob(os.path.join(BASE, tag, "*__*__*__*.parquet"))):
        paths[os.path.basename(p)] = p          # rerun overwrites full
rows = []
for p in sorted(paths.values(), key=os.path.basename):
    b = os.path.basename(p)[:-8].split("__")
    d = pd.read_parquet(p)
    rows.append(dict(model=b[0], dataset=b[1], strategy=b[2], mode=b[3], n=len(d),
                     acc=round(100 * (d.pred == d.gold).mean(), 2),
                     p_tok=round(d.prompt_tokens.mean(), 1) if "prompt_tokens" in d else None,
                     g_tok=round(d.gen_tokens.mean(), 1) if "gen_tokens" in d else None))
print(pd.DataFrame(rows).sort_values(["dataset", "mode", "model", "strategy"]).to_string(index=False))
