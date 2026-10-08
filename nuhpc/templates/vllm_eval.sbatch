# doc: Serve a model with vLLM on the node, run your client script against it, then shut down. Works for LLMs and the VLMs vLLM supports.
# doc: Params: model (HF id or local path), entry (client script in --code). Optional: vllm_args, startup_timeout (s, default 2700: ~19 min observed on Explorer, network storage), args.
# doc: The client gets OPENAI_BASE_URL / VLLM_URL (OpenAI-compatible API). Tensor parallel size = GPUs per node.
# doc: --pack: rows sharing model/vllm_args/startup_timeout/entry/args share one server (startup takes 2-19 min);
# doc: the client then runs once per row, writing to outputs/task_K/row_R. Size --time for all rows of a pack.
# pack-by: model vllm_args startup_timeout entry args
MODEL="$(hpc_param model)"
TIMEOUT="$(hpc_param startup_timeout 2700)"
ENTRY="$(hpc_param entry)"
EXTRA="$(hpc_param args "")"
VLLM_ARGS="$(hpc_param vllm_args "")"
PORT=$((8000 + RANDOM % 1000))   # shared nodes: avoid fixed ports
export VLLM_URL="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=EMPTY
# Explorer jobs inherit http_proxy (the route to the internet); localhost must bypass it, for curl and the client.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"

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
run_client() { python "$ENTRY" --params "$HPC_ROW_PARAMS" --out "$HPC_ROW_OUT" $EXTRA; }
hpc_each_row run_client
