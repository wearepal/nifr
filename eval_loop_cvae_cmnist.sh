#!/usr/bin/env bash

epochs="40"

function run_vae() {
        for scale in "0.0" "0.01" "0.02" "0.03" "0.04" "0.05"
        do
                echo $scale
                #qsub -pe smpslots 3 python.job start_vae.py 
                python start_vae.py \
			--dataset cmnist \
			--levels 4 \
			--enc-y-dim 64 \
			--lr 1e-3 \
			--disc-lr 1e-3 \
			--disc-enc-s-channels 256 \
			--super-val True \
			--init-channels 32 \
			--epochs 50 \
			--kl-weight 0.01 \
			--scale $scale "$@"
                sleep 1
        done
}

run_vae --results-csv more_evals.csv "$@"
