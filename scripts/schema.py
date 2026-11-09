import sys, os, zipfile, pandas as pd

ZIP = r"C:\Users\Mega-PC\Downloads\results_v2.zip"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(OUT)

files = sorted(f for f in os.listdir(OUT) if f.endswith(".parquet"))
print("parquet files:", len(files))
for name in ("Qwen2.5-14B__belebele__ar_direct.parquet",
             "Qwen2.5-14B__belebele__pivot_600m_seg.parquet",
             "belebele_translations_v2.parquet",
             "ammlu_translations_v2.parquet"):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        print("\n==", name, "MISSING"); continue
    df = pd.read_parquet(p)
    print("\n==", name, df.shape)
    print(df.dtypes.to_string())
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(df.head(2).to_string())
