import sys, os
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
rd = lambda n: pd.read_parquet(os.path.join(D, n))

for ds, N in (("belebele", 900), ("ammlu", 1500)):
    t = rd(f"{ds}_translations_v2.parquet")
    print("=" * 90)
    print(ds.upper(), " rows:", len(t))
    print("=" * 90)
    ok = t.block_ok_600m
    print("block_ok_600m: True %d (%.1f%%)  False %d" % (ok.sum(), 100 * ok.mean(), (~ok).sum()))
    gf = t.guard_flags
    print("guard_flags dtype:", gf.dtype)
    try:
        exploded = gf.explode().dropna()
        print("guard_flags value counts (exploded):")
        print(exploded.value_counts().head(12).to_string())
        print("rows with >=1 flag: %d (%.1f%%)" % (gf.map(len).gt(0).sum(), 100 * gf.map(len).gt(0).mean()))
    except Exception as e:
        print("  (not list-like:", e, ")")
        print(gf.value_counts().head(12).to_string())

    # empty / degenerate translations
    for c in [c for c in t.columns if c.startswith(("p_600m", "q_600m", "o_600m"))]:
        s = t[c].astype(str)
        print("   %-14s empty %4d   mean chars %7.1f" % (c, (s.str.strip() == "").sum(), s.str.len().mean()))

    # does translation failure explain accuracy loss?
    print("\n-- accuracy split by block_ok_600m (ctx strategy) --")
    for mdl in ("Qwen2.5-14B", "Qwen2.5-7B", "Llama-3.1-8B"):
        f = os.path.join(D, f"{mdl}__{ds}__pivot_600m_ctx.parquet")
        if not os.path.exists(f):
            continue
        r = rd(os.path.basename(f)).merge(t[["id", "block_ok_600m"]], on="id", how="left")
        r["correct"] = r.pred == r.gold
        g = r.groupby("block_ok_600m").correct.agg(["size", "mean"])
        g["mean"] = (100 * g["mean"]).round(2)
        print("  %-13s %s" % (mdl, g.to_dict("index")))

# english-vs-arabic length check on belebele, where human english exists
t = rd("belebele_translations_v2.parquet")
print("\n" + "=" * 90)
print("BELEBELE: human English vs 600m English, passage length (chars)")
print("=" * 90)
for c in ("p_ar", "p_en", "p_600m_seg", "p_600m_ctx", "p_600m_guard"):
    s = t[c].astype(str).str.len()
    print("   %-14s mean %7.1f  median %7.1f  min %5d  max %6d" % (c, s.mean(), s.median(), s.min(), s.max()))
ratio = t.p_600m_ctx.astype(str).str.len() / t.p_en.astype(str).str.len()
print("\n   600m_ctx / human_en length ratio: mean %.2f  median %.2f" % (ratio.mean(), ratio.median()))
print("   passages where 600m output < 50%% of human length: %d (%.1f%%)"
      % ((ratio < 0.5).sum(), 100 * (ratio < 0.5).mean()))
