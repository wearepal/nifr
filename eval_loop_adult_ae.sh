#!/bin/bash

topic="adult_ae"

for mix_fact in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do

python start_inn.py \
--ae-channels 61 \
--ae-enc-dim 35 \
--ae-epochs 60 \
--ae-levels 0 \
--ae-loss mixed \
--autoencode True \
--batch-norm False \
--batch-size 1000 \
--coupling-channels 20 \
--coupling-depth 1 \
--dataset adult \
--disc-hidden-dims 256 \
--disc-lr 0.0003 \
--drop-native True \
--epochs 200 \
--gamma 0.96 \
--glow True \
--input-noise False \
--level-depth 1 \
--lr 0.001 \
--nll-weight 10 \
--pred-s-weight 3e-3 \
--results-csv ${topic}.csv \
--scaling add2_sigmoid \
--super-val True \
--task-mixing-factor $mix_fact \
--train-on-recon False \
--vae False \
--val-freq 20 \
--weight-decay 1e-6 \
--zs-frac 0.03 \
"$@"

done
