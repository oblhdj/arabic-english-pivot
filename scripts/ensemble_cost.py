"""Two analyses that need no GPU and no API:

1. Agreement between the Arabic and pivot answers as a free confidence signal
   (reviewer weakness 5: the routing section has no working method).
2. Sensitivity of the token saving to the output:input price ratio
   (reviewer weakness 4: the fixed 4x weighting).
"""
import sys, os
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B, OUT = sys.argv[1], sys.argv[2]
rd = lambda *p: pd.read_parquet(os.path.join(B, *p))

PAIRS = [
    ("Qwen2.5-7B", "belebele", ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_direct__direct.parquet"),
     ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_mt_pivot__direct.parquet")),
    ("Llama-3.1-8B", "belebele", ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_direct__direct.parquet"),
     ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_mt_pivot__direct.parquet")),
    ("Qwen2.5-7B", "ammlu", ("v1rerun", "Qwen2.5-7B-Instruct__ammlu__ar_direct__direct.parquet"),
     ("v1full", "Qwen2.5-7B-Instruct__ammlu__ar_mt_pivot__direct.parquet")),
    ("Llama-3.1-8B", "ammlu", ("v1rerun", "Meta-Llama-3.1-8B-Instruct__ammlu__ar_direct__direct.parquet"),
     ("v1full", "Meta-Llama-3.1-8B-Instruct__ammlu__ar_mt_pivot__direct.parquet")),
    ("Qwen2.5-14B", "belebele", ("rv2", "Qwen2.5-14B__belebele__ar_direct.parquet"),
     ("rv2", "Qwen2.5-14B__belebele__pivot_600m_seg.parquet")),
    ("Qwen2.5-14B", "ammlu", ("rv2", "Qwen2.5-14B__ammlu__ar_direct.parquet"),
     ("rv2", "Qwen2.5-14B__ammlu__pivot_600m_seg.parquet")),
]

print("=" * 100)
print("1. AGREEMENT AS A CONFIDENCE SIGNAL")
print("   If the Arabic answer and the pivot answer match, how often is that answer right?")
print("=" * 100)
rows = []
for model, ds, fa, fp in PAIRS:
    a = rd(*fa)[["id", "gold", "pred"]].rename(columns={"pred": "pa"})
    p = rd(*fp)[["id", "pred"]].rename(columns={"pred": "pp"})
    m = a.merge(p, on="id")
    m = m[(m.pa >= 0) & (m.pp >= 0)]                 # both parsed
    ag = m.pa == m.pp
    acc_ag = (m.loc[ag, "pa"] == m.loc[ag, "gold"]).mean()
    acc_dis_ar = (m.loc[~ag, "pa"] == m.loc[~ag, "gold"]).mean()
    acc_dis_pv = (m.loc[~ag, "pp"] == m.loc[~ag, "gold"]).mean()
    rows.append(dict(model=model, dataset=ds, n=len(m),
                     agree_pct=round(100 * ag.mean(), 1),
                     acc_when_agree=round(100 * acc_ag, 1),
                     acc_disagree_arabic=round(100 * acc_dis_ar, 1),
                     acc_disagree_pivot=round(100 * acc_dis_pv, 1),
                     acc_arabic_overall=round(100 * (m.pa == m.gold).mean(), 1)))
A = pd.DataFrame(rows)
print(A.to_string(index=False))
print("\nReading: 'agree_pct' is the coverage you get for free. 'acc_when_agree' is")
print("precision on that slice. Compare it with 'acc_arabic_overall'.")
A.to_csv(OUT, index=False, encoding="utf-8")

print("\n" + "=" * 100)
print("2. OUTPUT-WEIGHT SENSITIVITY  (reviewer: 'weighted tokens assume 4:1')")
print("=" * 100)
MODES = [
    ("Qwen2.5-7B", "belebele", "direct", ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_direct__direct.parquet"),
     ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_mt_pivot__direct.parquet")),
    ("Qwen2.5-7B", "belebele", "cot", ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_direct__cot.parquet"),
     ("v1full", "Qwen2.5-7B-Instruct__belebele__ar_mt_pivot__cot.parquet")),
    ("Llama-3.1-8B", "belebele", "direct", ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_direct__direct.parquet"),
     ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_mt_pivot__direct.parquet")),
    ("Llama-3.1-8B", "belebele", "cot", ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_direct__cot.parquet"),
     ("v1full", "Meta-Llama-3.1-8B-Instruct__belebele__ar_mt_pivot__cot.parquet")),
]
print("%-13s %-6s %9s %9s   saving at w=1 / 4 / 10 / 50"
      % ("model", "mode", "out_share", "out_share"))
print("%-13s %-6s %9s %9s" % ("", "", "arabic", "pivot"))
for model, ds, mode, fa, fp in MODES:
    a, p = rd(*fa), rd(*fp)
    ai, ao = a.prompt_tokens.sum(), a.gen_tokens.sum()
    pi, po = p.prompt_tokens.sum(), p.gen_tokens.sum()
    sav = []
    for w in (1, 4, 10, 50):
        sav.append(100 * ((pi + w * po) - (ai + w * ao)) / (ai + w * ao))
    print("%-13s %-6s %8.1f%% %8.1f%%   %s"
          % (model, mode, 100 * ao / (ai + ao), 100 * po / (pi + po),
             "  ".join("%+6.1f%%" % s for s in sav)))
print("\nIn direct mode output is a rounding error, so the 4x choice cannot matter.")
print("In cot mode output dominates, so the assumption is load-bearing there only.")
print("\nwrote", OUT)
