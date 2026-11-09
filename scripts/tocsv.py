import sys, os, glob
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D, OUT = sys.argv[1], sys.argv[2]

rows = []
for p in sorted(glob.glob(os.path.join(D, "*__*__*.parquet"))):
    df = pd.read_parquet(p)
    model, dataset, strategy = os.path.basename(p)[:-8].split("__")
    m = df.pred >= 0
    rows.append(dict(
        model=model, dataset=dataset, strategy=strategy, n=len(df),
        accuracy_pct=round(100 * (df.pred == df.gold).mean(), 2),
        accuracy_parsed_only_pct=round(100 * (df.loc[m, "pred"] == df.loc[m, "gold"]).mean(), 2),
        unparsed=int((~m).sum()),
        mean_prompt_tokens=round(df.prompt_tokens.mean(), 1),
        total_prompt_tokens=int(df.prompt_tokens.sum()),
        mean_gen_tokens=round(df.gen_tokens.mean(), 2),
    ))

R = pd.DataFrame(rows).sort_values(["dataset", "model", "strategy"])

# attach baseline deltas where the baseline is present
base = {(r.model, r.dataset): r for r in R[R.strategy == "ar_direct"].itertuples()}
R["acc_delta_vs_ar"] = [
    round(r.accuracy_pct - base[(r.model, r.dataset)].accuracy_pct, 2)
    if (r.model, r.dataset) in base and r.strategy != "ar_direct" else None
    for r in R.itertuples()]
R["token_change_vs_ar_pct"] = [
    round(100 * (r.mean_prompt_tokens - base[(r.model, r.dataset)].mean_prompt_tokens)
          / base[(r.model, r.dataset)].mean_prompt_tokens, 1)
    if (r.model, r.dataset) in base and r.strategy != "ar_direct" else None
    for r in R.itertuples()]

R.to_csv(OUT, index=False, encoding="utf-8")
print("wrote", OUT, R.shape)
print(R.to_string(index=False))
