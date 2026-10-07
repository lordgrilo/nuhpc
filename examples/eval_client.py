"""Example client for the vllm_eval template: query the local vLLM server, write results to --out.

Contract used by every nuhpc template: --params <json> --out <dir>.
"""
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
p = json.load(open(a.params))
out = Path(a.out)

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key="EMPTY")
prompts = [json.loads(l) for l in open(p["prompts"])]   # e.g. {DATA}/prompts.jsonl, {"id":..,"prompt":..}
t0 = time.time()
with open(out / "completions.jsonl", "w") as f:
    for ex in prompts:
        r = client.chat.completions.create(
            model=p["model"], messages=[{"role": "user", "content": ex["prompt"]}],
            temperature=p.get("temperature", 0.0), max_tokens=p.get("max_tokens", 512))
        f.write(json.dumps({"id": ex["id"], "output": r.choices[0].message.content}) + "\n")
json.dump({"n": len(prompts), "seconds": time.time() - t0, **p}, open(out / "summary.json", "w"), indent=2)
