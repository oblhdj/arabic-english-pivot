"""Resumable Azure OpenAI runner for the Arabic-pivot prompt pack.

The API key is read from the AZURE_OPENAI_KEY environment variable, or from a
file given with --key-file. It is never printed, logged, or written to output.

  dry run   : python api_run.py --dry-run
  smoke test: python api_run.py --limit 10
  full run  : python api_run.py
"""
import argparse, json, os, re, sys, time
import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://pitchmind.openai.azure.com/openai/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
# USD per 1M tokens. VERIFY against the provider price sheet before trusting the cap.
# Gemini entries are 0.0 because the AI Studio free tier is not billed.
PRICES = {"gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "Phi-4": (0.125, 0.50),
          "gemini-3.5-flash": (0.0, 0.0), "gemini-3.1-flash-lite": (0.0, 0.0)}
ANS = re.compile(r"[1-5]")


def load_key(path):
    k = os.environ.get("AZURE_OPENAI_KEY", "").strip()
    if not k and path and os.path.exists(path):
        k = open(path, encoding="utf-8-sig").read().strip()
    return k


def parse_answer(text):
    """v1 convention: first digit 1-5 in the reply; unparsable counts as wrong."""
    m = ANS.search(text or "")
    return int(m.group()) if m else -1


def done_uids(path):
    if not os.path.exists(path):
        return set()
    s = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                s.add(json.loads(line)["uid"])
            except Exception:
                pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompt_pack/prompts_direct.jsonl")
    ap.add_argument("--out", default="api_results")
    ap.add_argument("--deployment", default="gpt-4.1")
    ap.add_argument("--cap-usd", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--key-file", default=r"C:\Users\Mega-PC\azure_key.txt")
    ap.add_argument("--provider", choices=("azure", "gemini"), default="azure")
    a = ap.parse_args()
    if a.provider == "gemini":
        a.base, a.auth, a.tokparam = GEMINI, "bearer", "max_tokens"
        a.effort = "none"          # Gemini 3.x reasons by default; direct mode must not
    else:
        a.base, a.auth, a.tokparam, a.effort = BASE, "api-key", "max_completion_tokens", None

    items = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    os.makedirs(a.out, exist_ok=True)
    outpath = os.path.join(a.out, f"{a.deployment}__results.jsonl")
    have = done_uids(outpath)
    todo = [x for x in items if x["uid"] not in have]
    if a.limit:
        todo = todo[:a.limit]

    pin, pout = PRICES.get(a.deployment, (2.00, 8.00))
    est_in = sum(len(x["prompt"]) for x in todo) / 3.5        # ~3.5 chars/token
    est = (est_in * pin + len(todo) * 8 * pout) / 1e6
    print("prompts file : %s (%d total)" % (a.prompts, len(items)))
    print("already done : %d" % len(have))
    print("to run       : %d" % len(todo))
    print("deployment   : %s   price in/out $%.3f/$%.3f per 1M" % (a.deployment, pin, pout))
    print("rough estimate: $%.2f   hard cap $%.2f" % (est, a.cap_usd))
    print("output       : %s" % outpath)

    key = load_key(a.key_file)
    print("api key      : %s" % ("found (%d chars)" % len(key) if key else "NOT FOUND"))
    if a.dry_run:
        by = {}
        for x in todo:
            by[(x["dataset"], x["strategy"])] = by.get((x["dataset"], x["strategy"]), 0) + 1
        print("\nbreakdown:")
        for k, v in sorted(by.items()):
            print("   %-9s %-12s %d" % (k[0], k[1], v))
        print("\nDRY RUN - no requests sent.")
        return
    if not key:
        sys.exit("No API key. Set AZURE_OPENAI_KEY or create %s" % a.key_file)

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    sess.headers["Authorization" if a.auth == "bearer" else "api-key"] = (
        ("Bearer " + key) if a.auth == "bearer" else key)
    url = f"{a.base}/chat/completions"
    spent, n_in, n_out, t0 = 0.0, 0, 0, time.time()

    with open(outpath, "a", encoding="utf-8") as fh:
        for i, x in enumerate(todo, 1):
            body = {"model": a.deployment,
                    "messages": [{"role": "user", "content": x["prompt"]}],
                    a.tokparam: x.get("max_new_tokens", 16),
                    "temperature": 0}
            if a.effort:
                body["reasoning_effort"] = a.effort
            for attempt in range(6):
                try:
                    r = sess.post(url, json=body, timeout=120)
                except requests.RequestException as e:
                    print("   net error (%s), retry %d" % (type(e).__name__, attempt + 1))
                    time.sleep(min(60, 2 ** attempt)); continue
                if r.status_code == 429:
                    wait = float(r.headers.get("retry-after", min(60, 2 ** attempt)))
                    print("   429, sleeping %.0fs" % wait); time.sleep(wait); continue
                if r.status_code >= 500:
                    time.sleep(min(60, 2 ** attempt)); continue
                break
            if r.status_code != 200:
                # never echo the key; the body may contain request detail only
                print("FAILED %s: HTTP %d %s" % (x["uid"], r.status_code, r.text[:300]))
                if r.status_code in (401, 403, 404):
                    sys.exit("Fatal HTTP %d - check deployment name and key." % r.status_code)
                continue
            d = r.json()
            txt = d["choices"][0]["message"]["content"]
            u = d.get("usage", {})
            ti, to = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            n_in += ti; n_out += to
            spent = (n_in * pin + n_out * pout) / 1e6
            fh.write(json.dumps({**{k: x[k] for k in ("uid", "dataset", "strategy", "mode", "id", "gold")},
                                 "model": a.deployment, "pred": parse_answer(txt),
                                 "output": txt, "prompt_tokens": ti, "completion_tokens": to},
                                ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(todo):
                fh.flush()
                rate = i / max(1e-9, time.time() - t0)
                print("   %d/%d  $%.3f  %.1f/s  eta %.0f min"
                      % (i, len(todo), spent, rate, (len(todo) - i) / rate / 60), flush=True)
            if spent > a.cap_usd:
                print("STOPPED: spend $%.2f exceeded cap $%.2f" % (spent, a.cap_usd))
                break

    print("\ndone. in=%d out=%d tokens, spend $%.3f, file %s" % (n_in, n_out, spent, outpath))


if __name__ == "__main__":
    main()
