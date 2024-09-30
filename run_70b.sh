#!/usr/bin/env bash
set -euo pipefail

# Download weights 
huggingface-cli download neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8


# Run benchmark 
cfg_file="configs/llama-3.1-70b-fp8-l40s.yaml"
workload_name=SyntheticWithSharedPrefix
python3 main.py \
    --engine-cfg-file $cfg_file \
    --workload-name $workload_name \
    --num-replicas 2
