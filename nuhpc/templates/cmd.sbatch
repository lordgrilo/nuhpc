# doc: Run any shell command from the --code snapshot, with no --params/--out contract: quick tests and one-off scripts. Params: cmd.
# doc: Output goes to the job log; files written to $HPC_OUT come back with `fetch`. `nuhpc run -- COMMAND` wraps this template.
# doc: With --rows-per-job, every row runs its own cmd, and $HPC_OUT is that row's own folder, so rows don't overwrite each other.
run_row() {
  local c
  c="$(hpc_param cmd)" || return 1
  cd "{{CODE}}"
  bash -c "$c"
}
hpc_each_row run_row
