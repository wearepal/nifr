#!/usr/bin/env bash

# ===========================================================
# ====================== Ablation for Zs ====================
# ===========================================================

MAX_SEED=1
frac_size=( 0.000082 0.000245 0.00041 0.00082 0.002442 0.00407 0.00814 0.02442 0.040691 )
gpu_id=0
slots=6

function run_nosinn() {
    for seed in $(seq $MAX_SEED); do
        echo $seed
        qsub -pe smpslots $slots python.job start_inn.py \
        --dataset celeba \
        --levels 3 \
        --level-depth 32 \
        --glow True \
        --reshape-method squeeze \
        --autoencode False \
        --input-noise True \
        --quant-level 5 \
        --use-wandb True \
        --factor-splits 0=0.5 1=0.5 \
        --train-on-recon False \
        --recon-detach False \
        --batch-size 32 \
        --nll-weight 1 \
        --pred-s-weight 1e-2 \
        --coupling-channels 512 \
        --super-val True \
        --super-val-freq 5000 \
        --val-freq 5000 \
        --task-mixing 0. \
        --gpu 0 \
        --num-discs 10 \
        --disc-channels 512 \
        --data-split-seed 42 \
        --save-dir experiments/ablation/zs/$seed "$@"
    done
}

for zs_frac in "${frac_size[@]}"; do
    run_nosinn --zs-frac $zs_frac "$@"
done
