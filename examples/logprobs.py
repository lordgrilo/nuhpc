"""Example script for the python template: score candidate answers by their log-probability under a model.

This is the direct mode: the model runs in-process with transformers, so everything inside the forward pass
is available. Here that is the exact log-probability of each forced continuation, which a vLLM server can't
return (it gives sampled text and the top-k log-probs).

Contract used by every nuhpc template: --params <json> --out <dir>.
params: model (HF id or path), prompts (jsonl rows {"id", "prompt", "choices": [...]}),
        optional dtype (default bfloat16), chat (true: wrap the prompt in the model's chat template).
Writes scores.jsonl ({"id", "logprobs": {choice: total log-prob}, "best"}) and summary.json. Resumable.
Choices are appended to the prompt as-is, so give them their leading space where the tokenizer expects one.
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
p = json.load(open(a.params))
out = Path(a.out)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(p["model"])
model = AutoModelForCausalLM.from_pretrained(p["model"], dtype=getattr(torch, p.get("dtype", "bfloat16")))
model = model.to(device).eval()


def head_ids(prompt: str) -> list[int]:
    if p.get("chat"):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       tokenize=False)
        return tok(text, add_special_tokens=False).input_ids
    return tok(prompt).input_ids


@torch.no_grad()
def score(prompt: str, choices: list[str]) -> list[float]:
    """Total log-prob of each choice as the continuation of prompt, all choices in one padded batch."""
    head = head_ids(prompt)
    seqs = [head + tok(c, add_special_tokens=False).input_ids for c in choices]
    ids = torch.full((len(seqs), max(map(len, seqs))), tok.pad_token_id or 0)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):   # right padding: a causal model's logits at the real positions never see it
        ids[i, :len(s)], mask[i, :len(s)] = torch.tensor(s), 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
    logp = torch.log_softmax(logits.float(), dim=-1)   # fp32: bf16 log-probs are too coarse to compare choices
    # position t predicts token t+1, so the choice tokens s[len(head):] are read at len(head)-1 .. len(s)-2
    return [logp[i, len(head) - 1:len(s) - 1].gather(-1, torch.tensor(s[len(head):], device=device)[:, None]).sum().item()
            for i, s in enumerate(seqs)]


rows = [json.loads(l) for l in open(p["prompts"]) if l.strip()]
path = out / "scores.jsonl"   # resumable: rows already written (by an interrupted run) are skipped
done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
t0 = time.time()
with open(path, "a") as f:
    for r in rows:
        if r["id"] in done:
            continue
        lp = score(r["prompt"], r["choices"])
        best = r["choices"][max(range(len(lp)), key=lp.__getitem__)]
        f.write(json.dumps({"id": r["id"], "logprobs": dict(zip(r["choices"], lp)), "best": best}) + "\n")
        f.flush()
json.dump({"n": len(rows), "skipped": len(done), "seconds": round(time.time() - t0, 2), "device": device, **p},
          open(out / "summary.json", "w"), indent=2)
