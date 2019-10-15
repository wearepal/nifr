#!/bin/bash

epochs="10"

for scale in "0.0" "0.01" "0.02" "0.03" "0.04" "0.05"
do
	python start.py \
		--train-on-recon False --level-depth 3 --levels 3 \
		--disc-channels 512 --coupling-depth 1 --coupling-channels 256 \
		--glow True --batch-norm False --spectral-norm False \
		--dataset cmnist --disc-lr 3e-4 --lr 3e-4 --gamma 1 \
		--use-comet True --nll-weight 1e-2 --base-density normal \
		--no-scaling False --data-pcnt 1 --val-freq 3 --super-val True \
		--scale 0.02 --eval-epochs 40 --batch-size 128 \
		--weight-decay 1e-6 --zs-frac 0.025 --padding 2 \
		--autoencode True --ae-epochs 5 --ae-levels 2 \
		--results-csv aelevels2_huber --epochs $epochs \
		--reshape-method squeeze --ae-loss huber --val-freq 1 \
		--scale $scale
done
