#!/usr/bin/bash

for MERGE_COUNT in "1" "2" "3" "4" "5"; do
    LINE_COUNT="2"
    for SEED in "0" "4" "5" "7" "8"; do
        echo "s${SEED}_m${MERGE_COUNT}_l${LINE_COUNT}"
        python3 ./experiments/measure_dfs_time.py --merge-count ${MERGE_COUNT} --line-count ${LINE_COUNT} --seed ${SEED} &
    done;
done;
wait