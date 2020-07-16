#!/usr/bin/env bash

epochs="40"

function run_baseline() {
	for scale in "0.0" "0.01" "0.02" "0.03" "0.04" "0.05"
	do
		echo $scale
		qsub -pe smpslots 3 python.job run_simple_baselines.py \
			--dataset cmnist \
			--lr 1e-3 \
			--batch-size 128 \
			--weight-decay 0 \
			--epochs $epochs \
			--padding 2 \
			--pred-s False \
			--scale $scale "$@"
		sleep 1
	done
}

run_baseline --greyscale False --results-csv more_metrics.csv "$@"
run_baseline --greyscale True --results-csv greyscale_more_metrics.csv "$@"
