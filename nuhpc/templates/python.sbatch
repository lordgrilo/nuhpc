# doc: Run a Python entry point from the --code snapshot. Params: entry (script path inside code dir); optional args (extra CLI string).
# doc: Contract: your script accepts --params <params.json> --out <dir>, and writes everything it wants kept to --out.
# Read params into variables first: a failing $(...) only aborts the job under set -e in an assignment.
ENTRY="$(hpc_param entry)"
EXTRA="$(hpc_param args "")"
cd "{{CODE}}"
python "$ENTRY" --params "$HPC_PARAMS" --out "$HPC_OUT" $EXTRA
