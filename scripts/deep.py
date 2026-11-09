import sys, os
from math import comb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
rd = lambda n: pd.read_parquet(os.path.join(D, n))


def mcnemar(b, c):
    """Exact two-sided McNemar. b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ---------- 1. item-level agreement, Qwen14 (the only model with baselines here)
print("=" * 95)
print("ITEM-LEVEL AGREEMENT vs ar_direct  (Qwen2.5-14B)")
print("=" * 95)
for ds in ("belebele", "ammlu"):
    b = rd(f"Qwen2.5-14B__{ds}__ar_direct.parquet")[["id", "gold", "pred"]]
    for strat in ("en_human", "pivot_600m_ctx", "pivot_600m_seg"):
        f = f"Qwen2.5-14B__{ds}__{strat}.parquet"
        if not os.path.exists(os.path.join(D, f)):
            continue
        o = rd(f)[["id", "pred"]]
        m = b.merge(o, on="id", suffixes=("_base", "_alt"))
        bc = m.pred_base == m.gold
        oc = m.pred_alt == m.gold
        both = int((bc & oc).sum()); nb = int((bc & ~oc).sum())
        no = int((~bc & oc).sum()); neither = int((~bc & ~oc).sum())
        p = mcnemar(nb, no)
        print("\n%s / %s  (n=%d)" % (ds, strat, len(m)))
        print("   both correct %4d | only ar_direct %4d | only %s %4d | both wrong %4d"
              % (both, nb, strat, no, neither))
        print("   recovered %d items the Arabic baseline missed; lost %d it had" % (no, nb))
        print("   McNemar exact p = %.3g %s" % (p, "(significant)" if p < 0.05 else "(not significant)"))

# ---------- 2. ArabicMMLU per-subject
print("\n" + "=" * 95)
print("ARABICMMLU PER-SUBJECT: where the 600m pivot collapses (Qwen2.5-14B)")
print("=" * 95)
t = rd("ammlu_translations_v2.parquet")[["id", "subject", "group", "guard_flags", "block_ok_600m"]]
b = rd("Qwen2.5-14B__ammlu__ar_direct.parquet")[["id", "gold", "pred"]].rename(columns={"pred": "pred_ar"})
o = rd("Qwen2.5-14B__ammlu__pivot_600m_ctx.parquet")[["id", "pred"]].rename(columns={"pred": "pred_pv"})
m = t.merge(b, on="id").merge(o, on="id")
m["ok_ar"] = m.pred_ar == m.gold
m["ok_pv"] = m.pred_pv == m.gold
g = m.groupby("subject").agg(n=("id", "size"), ar=("ok_ar", "mean"), pv=("ok_pv", "mean"),
                             flags=("guard_flags", lambda s: (s > 0).mean()))
g[["ar", "pv", "flags"]] = (100 * g[["ar", "pv", "flags"]]).round(1)
g["delta"] = (g.pv - g.ar).round(1)
g = g[g.n >= 20].sort_values("delta")
print("(flags%% = share of items the translation guard flagged)")
print(g.to_string())

print("\nby group:")
gg = m.groupby("group").agg(n=("id", "size"), ar=("ok_ar", "mean"), pv=("ok_pv", "mean"))
gg[["ar", "pv"]] = (100 * gg[["ar", "pv"]]).round(1)
gg["delta"] = (gg.pv - gg.ar).round(1)
print(gg.sort_values("delta").to_string())

if len(sys.argv) > 2:
    out = g.rename(columns={"ar": "ar_direct_acc_pct", "pv": "pivot_600m_ctx_acc_pct",
                            "flags": "guard_flagged_pct", "delta": "acc_delta_pts"})
    out.to_csv(sys.argv[2], encoding="utf-8")
    print("\nwrote", sys.argv[2])

# ---------- 3. guard flags vs accuracy
print("\n" + "=" * 95)
print("DOES THE GUARD FLAG PREDICT FAILURE?  pivot_600m_ctx accuracy by flag count")
print("=" * 95)
for ds in ("belebele", "ammlu"):
    tt = rd(f"{ds}_translations_v2.parquet")[["id", "guard_flags"]]
    print("\n--", ds)
    for mdl in ("Qwen2.5-14B", "Qwen2.5-7B", "Llama-3.1-8B"):
        f = f"{mdl}__{ds}__pivot_600m_ctx.parquet"
        if not os.path.exists(os.path.join(D, f)):
            continue
        r = rd(f)[["id", "gold", "pred"]].merge(tt, on="id")
        r["ok"] = r.pred == r.gold
        s = r.groupby(r.guard_flags.clip(upper=2)).ok.agg(["size", "mean"])
        s["mean"] = (100 * s["mean"]).round(1)
        print("   %-13s" % mdl, {int(k): (int(v["size"]), v["mean"]) for k, v in s.iterrows()})
print("\n   key: {flag_count: (n, accuracy%)}, flag counts clipped at 2+")
