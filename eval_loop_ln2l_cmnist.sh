#!/usr/bin/env bash

epochs="40"

function run_baseline() {
	for scale in "0.0" "0.01" "0.02" "0.03" "0.04" "0.05"
	do
		echo $scale
		qsub -pe smpslots 3 python.job -m nsfiair.baselines.ln2l_cmnist \
			--use-ln2l-data False \
			--scale $scale "$@"
		sleep 1
	done
}

run_baseline --results-csv more_metrics.csv "$@"
