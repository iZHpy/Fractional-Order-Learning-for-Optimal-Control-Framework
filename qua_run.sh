#!/bin/bash

seeds=(0 1 1024 2025 42)

commands=()

for seed in "${seeds[@]}"
do
    logs_dir="./logs/qua/$seed/"
    commands+=("python3 main.py --seed $seed --logs $logs_dir")
done

echo "Total commands: ${#commands[@]}"

printf "%s\n" "${commands[@]}" | xargs -P 5 -I {} bash -c "{}"