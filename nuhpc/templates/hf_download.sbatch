# doc: Download a Hugging Face model/dataset snapshot into HF_HOME on cluster storage so GPU jobs can run offline. A CPU profile is enough.
# doc: Params: repo (e.g. Qwen/Qwen2.5-VL-7B-Instruct). Optional: repo_type (model|dataset), revision. Gated repos need HF_TOKEN in env_setup.
export HF_HUB_OFFLINE=0
python - <<'PY'
import os, json
from huggingface_hub import snapshot_download
p = json.load(open(os.environ["HPC_PARAMS"]))
path = snapshot_download(p["repo"], repo_type=p.get("repo_type", "model"), revision=p.get("revision"))
print("downloaded to", path)
json.dump({"repo": p["repo"], "path": path}, open(os.path.join(os.environ["HPC_OUT"], "result.json"), "w"))
PY
