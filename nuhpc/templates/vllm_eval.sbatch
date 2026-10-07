# doc: Serve a model with vLLM on the node, run your client script against it, then shut down. Works for LLMs and the VLMs vLLM supports.
# doc: Params: model (HF id or local path), entry (client script in --code). Optional: vllm_args, startup_timeout (s, default 1200), args.
# doc: The client gets OPENAI_BASE_URL / VLLM_URL (OpenAI-compatible API). Tensor parallel size = GPUs per node.
MODEL="$(hpc_param model)"
TIMEOUT="$(hpc_param startup_timeout 1200)"
ENTRY="$(hpc_param entry)"
EXTRA="$(hpc_param args "")"
VLLM_ARGS="$(hpc_param vllm_args "")"
PORT=$((8000 + RANDOM % 1000))   # shared nodes: avoid fixed ports
export VLLM_URL="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=EMPTY

vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "{{GPUS_PER_NODE}}" $VLLM_ARGS \
  > "$HPC_OUT/vllm_server.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

waited=0
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "[nuhpc] vLLM exited during startup:"; tail -50 "$HPC_OUT/vllm_server.log"; exit 1; fi
  if [ "$waited" -ge "$TIMEOUT" ]; then echo "[nuhpc] vLLM not healthy after ${TIMEOUT}s"; tail -50 "$HPC_OUT/vllm_server.log"; exit 1; fi
  sleep 10; waited=$((waited + 10))
done
echo "[nuhpc] vLLM up after ${waited}s at $VLLM_URL"

cd "{{CODE}}"
python "$ENTRY" --params "$HPC_PARAMS" --out "$HPC_OUT" $EXTRA
