"""Example VLM client for the vllm_eval template: send image + text prompts to the local vLLM server.

Prompts file (JSONL): {"id": ..., "prompt": "...", "image": "<path or http(s) URL>"} per line.
Relative image paths resolve against the prompts file's directory; local images are sent inline as base64.
Contract used by every nuhpc template: --params <json> --out <dir>.
"""
import argparse, base64, json, mimetypes, os, time
from pathlib import Path
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
p = json.load(open(a.params))
out = Path(a.out)
prompts_path = Path(p["prompts"])   # e.g. {DATA}/my-eval/prompts.jsonl, images next to it


def image_url(ref: str) -> str:
    if ref.startswith(("http://", "https://", "data:")):
        return ref   # remote URLs are fetched by the vLLM server (compute nodes have internet)
    f = prompts_path.parent / ref
    mime = mimetypes.guess_type(f.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()


client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key="EMPTY")
rows = [json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip()]
t0 = time.time()
path = out / "completions.jsonl"   # resumable: rows already written (by an interrupted run) are skipped
done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
with open(path, "a") as f:
    for ex in rows:
        if ex["id"] in done:
            continue
        content = [{"type": "image_url", "image_url": {"url": image_url(ex["image"])}},
                   {"type": "text", "text": ex["prompt"]}]
        r = client.chat.completions.create(
            model=p["model"], messages=[{"role": "user", "content": content}],
            temperature=p.get("temperature", 0.0), max_tokens=p.get("max_tokens", 256))
        f.write(json.dumps({"id": ex["id"], "output": r.choices[0].message.content}) + "\n")
json.dump({"n": len(rows), "seconds": time.time() - t0, **p}, open(out / "summary.json", "w"), indent=2)
