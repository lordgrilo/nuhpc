# doc: Environment check on a compute node: GPUs, Python/torch/vllm imports, outbound internet, HF cache, disk. Run this first.
echo "== node: $(hostname)"
nvidia-smi || echo "no GPU visible"
echo "== python: $(command -v python) $(python --version 2>&1)"
python - <<'PY' || true
for mod in ("torch", "transformers", "vllm"):
    try:
        m = __import__(mod); print(f"{mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod}: NOT importable ({type(e).__name__})")
try:
    import torch; print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
except Exception:
    pass
PY
echo "== HF_HOME=${HF_HOME:-unset} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" https://huggingface.co || true)
if [ -n "$code" ] && [ "$code" != "000" ]; then echo "== internet from compute node: YES ($code)"
else echo "== internet from compute node: NO -> pre-download weights (hf_download where internet works, or nuhpc push)"; fi
echo "== disk"; df -h "{{REMOTE_ROOT}}" || true
echo '{"smoke": "ok"}' > "$HPC_OUT/result.json"
