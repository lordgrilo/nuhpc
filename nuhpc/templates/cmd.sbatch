# doc: Run any shell command from the --code snapshot, with no --params/--out contract: quick tests and one-off scripts. Params: cmd.
# doc: Output goes to the job log; files written to $HPC_OUT come back with `fetch`. `nuhpc run -- COMMAND` wraps this template.
CMD="$(hpc_param cmd)"
cd "{{CODE}}"
bash -c "$CMD"
