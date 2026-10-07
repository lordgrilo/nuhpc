# doc: PyTorch distributed training via torchrun (1..N nodes, all GPUs per node). Params: entry (script in --code); optional args.
# doc: Your script uses torch.distributed as usual (env:// init) and gets --params/--out. Only rank 0 should write to --out.
ENTRY="$(hpc_param entry)"
EXTRA="$(hpc_param args "")"
cd "{{CODE}}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
srun --ntasks-per-node=1 torchrun \
  --nnodes="{{NODES}}" --nproc-per-node="{{GPUS_PER_NODE}}" \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv-id="$SLURM_JOB_ID" \
  "$ENTRY" --params "$HPC_PARAMS" --out "$HPC_OUT" $EXTRA
