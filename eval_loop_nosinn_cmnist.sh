#!/bin/bash

scales=( 0 0.01 0.02 0.03 0.04 0.05 )

function run_cmnist() {
	for scale in "${scales[@]}"; do
		echo "scale=${scale}"
		qsub -pe smpslots 6 python.job start_inn.py \
		--dataset cmnist \
		--levels 3 \
		--level-depth 24 \
		--glow True \
		--reshape-method squeeze \
		--autoencode False \
		--input-noise True \
		--quant-level 5 \
		--use-wandb True \
		--factor-splits 0=0.5 1=0.5 \
		--train-on-recon False \
		--recon-detach False \
		--batch-size 256 \
		--test-batch-size 512 \
		--nll-weight 1 \
		--pred-s-weight 1e-2 \
		--zs-frac 0.002 \
		--coupling-channels 512 \
		--super-val True \
		--super-val-freq 5_000 \
		--num-discs 1 \
		--early-stop 10000 \
		--disc-channels 512 \
		--level-depth 24 \
		--num-discs 3 \
		--iter 26_000 \
		--scale $scale "$@"
		sleep 1
	done
}

run_cmnist --gpu 0 --results-csv more_eval.csv "$@"
