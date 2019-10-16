#!/usr/bin/env bash

epochs="40"

for greyscale in "False" "True"
do
for scale in "0.0" "0.01" "0.02" "0.03" "0.04" "0.05"
do
	python naive.py \
		--dataset cmnist --scale $scale --greyscale $greyscale \
		--lr 1e-3 --batch-size 128 --weight-decay 0 --epochs $epochs \
		--padding 2 --pred-s False
done
done
