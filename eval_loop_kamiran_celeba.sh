#!/usr/bin/env bash

epochs="40"

for mf in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
do
	python run_simple_baselines.py \
		--dataset celeba --task-mixing-factor $mf \
		--lr 1e-3 --batch-size 256 --weight-decay 0 --epochs $epochs \
		--pred-s False --method kamiran $@
done

