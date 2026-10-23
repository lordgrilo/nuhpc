# doc: Measure how much space a folder tree uses, in parallel and read-only: the job behind `nuhpc disk --scan`. Params: root (an absolute path), depth (levels listed, default 4), names (false: record no file names), workers (default 16), budget (seconds before unread folders are left out).
# doc: Writes outputs/task_0/disk.json (sizes, file counts and newest change per folder, plus the largest files); `nuhpc disk` fetches and shows it. Nothing is changed or deleted.
# env-setup: none
NAMES=""
[ "$(hpc_param names true)" = "true" ] || NAMES="--no-names"
python3 "{{CODE}}/diskscan.py" --root "$(hpc_param root)" --depth "$(hpc_param depth 4)" --workers "$(hpc_param workers 16)" \
  --budget "$(hpc_param budget 1500)" $NAMES --out "$HPC_OUT/disk.json"
