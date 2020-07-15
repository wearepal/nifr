#!/usr/bin/env bash

epochs=( 10 20 30 40 50 60 )
paths=( 1594628957.05065,1 1594628956.8866942,3 1594628953.1033692,5 1594628953.53084,10 1594628954.4587193,30 1594628954.4766445,50 1594628953.123214,100 1594628956.1824822,300 1594628957.1935444,500 )
slots=6

for i in ${paths[@]}; do
    IFS=",";
    set $i;
    for epoch in ${epochs[@]}; do
        qsub -pe smpslots $slots python.job do_evaluation.py "/mnt/data/tk324/NoSINN/experiments/ablation/zs/1/${1}/checkpt_epoch${epoch}.pth" --results-csv "zsdim${2}_epoch${epoch}.csv";
        sleep 1;
    done
done
