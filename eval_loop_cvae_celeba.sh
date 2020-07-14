#!/bin/bash

# for mix_fact in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
for mix_fact in "0.0" "1.0"
do
python start_vae.py \
    --levels 5 \
    --elbo-weight 1e-2 \
    --kl-weight 1 \
    --recon-loss l1 \
    --pred-s-weight 1 \
    --super-val True \
    --epochs 0 \
    --dataset celeba \
    --eval-epochs 40 \
    --task-mixing-factor $mix_fact \
    --use-wandb False \
    "$@"
done
