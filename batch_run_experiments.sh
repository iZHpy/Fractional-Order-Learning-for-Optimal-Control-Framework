#!/bin/bash

folder_path="./data/cartpole/"
seeds=(1 1024 2025 42)
seed=${seeds[3]}

mapfile -t dirs < <(find "$folder_path" -mindepth 1 -type d)

# for seed in "${seeds[@]}"
# do
for dir in "${dirs[@]}"
do
    sub_dir=${dir#./data/}
    logs_dir="./logs/$sub_dir"
    mkdir -p "$logs_dir"
    # exec python3 main.py --seed $seed --data_dir $dir --logs $logs_dir
    commands+=("python3 main.py --seed $seed --data_dir $dir --logs $logs_dir")
done
# done


# 使用 xargs 并行执行命令
printf "%s\n" "${commands[@]}" | xargs -P 1 -I {} bash -c "{}"
