#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sample a PTD student of MiniMax-H3 on NPROC GPUs (default 4); GPU r renders every prompt with seed r:
#
#   bash scripts/sample.sh --recipe h3_6step --ckpt weights/ptd_h3_6step --model models/MiniMax-H3 \
#       --prompts prompts.json --cond-cache cond_cache --output samples/h3_6step
#
# Text prompts missing from the condition cache are encoded first (scripts/encode_prompts.py, one process per GPU);
# image-conditioned requests must be encoded beforehand with scripts/encode_image_conditions.py.
# All arguments go to ptd/sample.py (python -m ptd.sample --help).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
NPROC=${NPROC:-4}

# Pick out the arguments the condition-cache encoder needs.
RECIPE="" CACHE="" MODEL=() PROMPTS=()
ARGS=("$@")
i=0
while [ "$i" -lt "${#ARGS[@]}" ]; do
  a="${ARGS[i]}"
  case "$a" in
    --recipe) RECIPE="${ARGS[i + 1]:-}" ;;
    --recipe=*) RECIPE="${a#*=}" ;;
    --cond-cache) CACHE="${ARGS[i + 1]:-}" ;;
    --cond-cache=*) CACHE="${a#*=}" ;;
    --model) MODEL=(--model "${ARGS[i + 1]:-}") ;;
    --model=*) MODEL=(--model "${a#*=}") ;;
    --prompts)
      j=$((i + 1))
      while [ "$j" -lt "${#ARGS[@]}" ] && [[ "${ARGS[j]}" != --* ]]; do PROMPTS+=("${ARGS[j]}"); j=$((j + 1)); done ;;
  esac
  i=$((i + 1))
done
if [[ "$RECIPE" == h3_* && -n "$CACHE" ]]; then
  TODO=$(python "$REPO/scripts/encode_prompts.py" --prompts "${PROMPTS[@]}" --cache-dir "$CACHE" --list-only | grep -o "todo [0-9]*" | cut -d" " -f2)
  if [ "${TODO:-0}" -gt 0 ]; then
    NG=$(( TODO < NPROC ? TODO : NPROC ))
    for i in $(seq 0 $((NG - 1))); do
      CUDA_VISIBLE_DEVICES=$i python "$REPO/scripts/encode_prompts.py" --prompts "${PROMPTS[@]}" --cache-dir "$CACHE" "${MODEL[@]}" \
        --rank "$i" --world "$NG" &
    done
    wait
  fi
fi
exec torchrun --nnodes=1 --standalone --nproc_per_node="$NPROC" -m ptd.sample "$@"
