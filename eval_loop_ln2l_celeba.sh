#!/usr/bin/env bash

function run_ln2l() {
	# for mf in "0.0" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "0.95" "1.0"
	for mf in "0.0" "1.0"
	do
		echo $mf
		qsub -pe smpslots 3 python.job start_ln2l.py \
			--dataset celeba \
			--task-mixing-factor $mf \
			--lr 1e-3 \
			--batch-size 128 \
			--weight-decay 0 \
			--epochs 40 \
			--entropy-weight 0.01 "$@"
		sleep 1
	done
}

run_ln2l --celeba-sens-attr Male Young --results-csv agender.csv "$@"
run_ln2l --celeba-sens-attr Black_Hair Blond_Hair Brown_Hair --results-csv hair_color.csv "$@"
