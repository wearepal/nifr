#!/usr/bin/env bash

epochs="40"

#for mf in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
for mf in "0.0" "1.0"
do
	python run_simple_baselines.py \
		--dataset celeba --task-mixing-factor $mf \
		--lr 1e-3 --batch-size 128 --weight-decay 0 --epochs $epochs \
		--pred-s False "$@"
done

