# Token cost and accuracy of English pivoting for Arabic LLM queries

Code, prompts and per-question results for the paper *Thinking in English,
Paying Less? Token Cost and Accuracy of English Pivoting for Arabic LLM
Queries*.

The question is whether translating an Arabic query into English before
sending it to a language model saves enough tokens to be worth the accuracy it
costs. The short answer is that the saving is predictable from the tokenizer
and the cost is predictable from the model's Arabic ability, and the two pull
in opposite directions.

## What is here

`paper/` holds the LaTeX source and the compiled PDF.

`prompts/` holds every prompt that was sent to a model, one JSON object per
line. `prompts_direct.jsonl` has 5,700 prompts across five conditions
(Belebele Arabic, pivot and human English; ArabicMMLU Arabic and pivot) and
`prompts_cot.jsonl` has 900 step-by-step prompts for Belebele.
`manifest.json` gives a SHA-256 digest per condition, so a rerun can be checked
against the original byte for byte. The prompts carry no chat template; each
model's own template is applied at call time.

`data/` holds the raw per-question results. Every row has the question id, the
gold answer, the parsed prediction, the model's full reply and the token counts
the model or API reported. The three Kaggle runs are in `v1/`, `v2/` and `v3/`
as Parquet; the two API models are in `api/` as JSONL.

`scripts/` holds the runner and the analysis code. `api_run.py` queries Azure
OpenAI or Gemini, resumes from a partial run by question id, and stops at a
spending cap. The rest read the result files and produce the tables in the
paper.

`summaries/` holds those tables as CSV, which is the quickest way to check a
number in the paper against the data behind it.

## Reproducing the runs

The three Kaggle notebooks are in `notebooks/`. Each needs a GPU T4 x2 session
with internet on. The first produces `results.zip` and `rerun_results.zip`, the
second `results_v2.zip` and the third `results_v3.zip`. Unzip each into the
matching directory under `data/`.

The API runs need an Azure AI Foundry key with Phi-4 and GPT-4.1 deployed. Put
the key in the `AZURE_OPENAI_KEY` environment variable, then:

```
python scripts/api_run.py --dry-run
python scripts/api_run.py --deployment gpt-4.1 --cap-usd 8
python scripts/api_run.py --deployment Phi-4 --cap-usd 8
```

The dry run reports how many prompts are left and what they should cost before
anything is sent. Both models together came to $3.26 at November 2026 prices.
Expect a few refusals from the content filter: 20 of 5,700 for GPT-4.1 and 14
for Phi-4, which are reported but not retried, since they fail the same way
every time.

Step-by-step answers need a second pass, because the runner takes the first
digit in the reply and a chain of thought contains digits before it reaches the
answer:

```
python scripts/api_run.py --prompts prompts/prompts_cot.jsonl --out data/api_cot
python scripts/reparse_cot.py data/api_cot summaries/cot_pivot_effect.csv
```

## Reproducing the analysis

Each script takes the directory holding the extracted data and writes one CSV.
They need pandas and pyarrow and nothing else.

```
python scripts/analyze.py data summaries/results_v2_summary.csv
python scripts/replicate.py data summaries/subject_replication.csv
python scripts/v3_compare.py data summaries/v3_gap_closed.csv
python scripts/api_analyze.py data summaries/api_pivot_effect.csv
python scripts/ensemble_cost.py data summaries/agreement_signal.csv
```

`v3_compare.py` is the one behind the central claim about translator size. It
pairs the 600M and 3.3B results per question and reports an exact McNemar test
plus the fraction of the distance to the human English reference that the
larger translator recovers.

## Known gaps

ArabicMMLU has no human English version, so the gap-closed figures for it use
the Arabic baseline as the ceiling instead. Phi-4 ignored the 16-token output
cap on 29.7% of calls, so its output token counts are not comparable with the
other models' and its savings are computed on prompt tokens only. Everything
uses greedy decoding with one seed. Two Qwen2.5-14B guard runs were never
scheduled, which is why that strategy has four rows and not six.

## Data licence

Belebele and ArabicMMLU keep their own licences. NLLB-200 is CC-BY-NC, so the
translations here are non-commercial. The code is MIT.

## Citation

```
@inproceedings{benelhadj2026pivot,
  title     = {Thinking in English, Paying Less? Token Cost and Accuracy
               of English Pivoting for Arabic LLM Queries},
  author    = {Ben Elhadj, Oussema},
  year      = {2026}
}
```
