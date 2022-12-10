#!/usr/bin/bash

LINE_COUNTS_SMALL="250 500 750 1000 1250 1500 1750 2000"

# for model in "faster_notok" "faster_tok" "slow_tok" "slow_notok";
# do
#     ./experiments/measure_time.py \
#         -m "${model}" \
#         -l "computed/time_${model}.jsonl" \
#         -n ${LINE_COUNTS_SMALL} \
#         -v 1000 \
#     ;
# done;

LINE_COUNTS_LARGE="50000 100000 500000 1000000"
# for model in "faster_notok" "faster_tok";
for model in "sentencepiece";
do
    ./experiments/measure_time.py \
        -m "${model}" \
        -l "computed/time_${model}.jsonl" \
        -n ${LINE_COUNTS_SMALL} ${LINE_COUNTS_LARGE} \
        -v 8000 \
    ;
done;