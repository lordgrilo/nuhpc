# doc: Run a Python entry point from the --code snapshot. Params: entry (script path inside code dir); optional args (extra CLI string).
# doc: Contract: your script accepts --params <params.json> --out <dir>, and writes everything it wants kept to --out.
# doc: With --rows-per-job, every row runs this same command with its own params and its own --out folder (rows may differ in entry and args).
run_row() {
  local entry extra
  # Read params into variables first, and stop on a missing one: a failing $(...) inside a command would not.
  entry="$(hpc_param entry)" || return 1
  extra="$(hpc_param args "")" || return 1
  cd "{{CODE}}"
  python "$entry" --params "$HPC_PARAMS" --out "$HPC_OUT" $extra
}
hpc_each_row run_row
