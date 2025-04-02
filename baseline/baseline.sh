#!/bin/bash

seeds=(0 1 1024 2025 42)
Ns=(16 32)
commands=()

for seed in "${seeds[@]}"
do
    logs_dir="./logs/PID/cartpole"
    # data_dir="./data/quadrotor"
    mkdir -p "$logs_dir"
    nohup python3 cartpole_PIDbaseline.py --seed $seed > $logs_dir/log_$seed.txt 2>&1 &
    # nohup python3 Baseline_FODS.py --seed $seed --dim 16 --N $N > "$logs_dir/log_$seed.txt" 2>&1 &
    
done



# for seed in "${seeds[@]}"
# do
#     for N in "${Ns[@]}"
#     do
#         logs_dir="./logs/baseline/dim16/$N"
#         mkdir -p "$logs_dir"
#         # nohup python3 Baseline_FODS.py --seed $seed --dim 16 --N $N > "$logs_dir/log_$seed.txt" 2>&1 &
#         # exec "python3 Baseline_FODS.py --seed $seed --dim 16 --N $N > $logs_dir/log_$seed.txt 2>&1 &"
#     done
# done

# do
#     logs_dir="./logs/baseline/dim64/"
#     mkdir -p "$logs_dir"
#     nohup python3 Baseline_FODS.py --seed $seed --dim 64 > "$logs_dir/log_$seed.txt" 2>&1 &
# done
