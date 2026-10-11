# doc: Build your own Python environment for nuhpc jobs, once (about 20 min on a CPU profile): uv, Python 3.12, torch and vLLM.
# doc: Optional params: env (default envs/llm, relative to remote_root unless absolute), python (3.12), torch_backend (cu129),
# doc: vllm (wheel URL or pip spec, or none), extra (more packages; default "huggingface_hub openai"), bootstrap_python (a python with pip).
# doc: The example config's env_setup already activates $HPC_REMOTE_ROOT/envs/llm. Then run `nuhpc check --smoke`.
# env-setup: none
ENV="$(hpc_param env envs/llm)"
PYVER="$(hpc_param python 3.12)"
BACKEND="$(hpc_param torch_backend cu129)"
VLLM="$(hpc_param vllm https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl)"
EXTRA="$(hpc_param extra "huggingface_hub openai")"
BOOT="$(hpc_param bootstrap_python /shared/EL9/explorer/anaconda3/2024.06/bin/python3)"
case "$ENV" in /*) ;; *) ENV="{{REMOTE_ROOT}}/$ENV" ;; esac

# Project storage costs ~0.1 s per file written: uv's cache stays on local disk, files are copied (no
# cross-filesystem hardlinks), and bytecode is compiled now (first imports would otherwise write each .pyc).
export UV_CACHE_DIR="/tmp/$USER-uv-$HPC_RUN_ID" UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
export UV_PYTHON_INSTALL_DIR="$(dirname "$ENV")/.python"   # the interpreter must be on storage every node sees
UVBIN="/tmp/$USER-uvbin-$HPC_RUN_ID"
trap 'rm -rf "$UV_CACHE_DIR" "$UVBIN"' EXIT

# uv itself comes from PyPI, through a python that has pip (Explorer's system python3 has none).
[ -x "$BOOT" ] || BOOT=python3
if ! "$BOOT" -m pip --version >/dev/null 2>&1; then
  echo "[nuhpc] $BOOT has no pip: pass -p bootstrap_python=<path to a python with pip>"; exit 1
fi
"$BOOT" -m pip install -q --target "$UVBIN" uv
UV="$UVBIN/bin/uv"

if [ -x "$ENV/bin/python" ]; then
  echo "[nuhpc] $ENV exists: installing into it. To start clean, build at a new env path."
else
  "$UV" venv --python "$PYVER" "$ENV"
fi
PKGS=()
case "$VLLM" in none) ;; http*) PKGS+=("vllm @ $VLLM") ;; *) PKGS+=("$VLLM") ;; esac
set -f; PKGS+=($EXTRA); set +f   # word-split the list, but never glob specs such as transformers[torch]
if [ ${#PKGS[@]} -gt 0 ]; then
  "$UV" pip install --python "$ENV/bin/python" --torch-backend="$BACKEND" "${PKGS[@]}"
fi

"$ENV/bin/python" - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for name in ("torch", "transformers", "vllm", "huggingface_hub", "openai"):
    try:
        print(name, getattr(importlib.import_module(name), "__version__", "?"))
    except ImportError:
        print(name, "not installed")
PY
echo "[nuhpc] built $ENV. env_setup needs: source \"$ENV/bin/activate\""
