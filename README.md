# No-Shortcuts in Neural Networks

## Requirements

Python 3.6 or higher

## Installing dependencies

The dependencies are listed in the `setup.py`.
To install them all, do

```
pip install -e .
```

## Running the code

For the UCI Adult dataset use `eval_loop_adult_ae_on_recon.sh`.

To run on CMNIST use `eval_loop_nosinn_cmnist.sh`.

Our results on CelebA can be reconstructed with:

```
python start.py --dataset celeba --levels 3 --level-depth 8 --reshape-method haar --glow False \
        --batch-norm False --coupling-channels 512 --val-freq 10 --autoencode True --use-wandb True \
        --zs-frac 0.05 --path-to-ae checkpt_giddy-dew-373.pth --quant-level 8 --input-noise False \
        --nll-weight 1e2 --ae-levels 3 --ae-channels 64 --ae-enc-dim 16 --vae True \
        --train-on-recon False --pred-s-weight 1 --disc-channels 512 --lr 3e-4 --disc-lr 3e-4 \
        --weight-decay 1e-5 --epochs 30 --super-val True --eval-epochs 5 --val-freq 5 \
        --disc-channels 512 --train-on-recon True --pred-s-weight 1 --recon-stability 1 \
        --input-noise False
```
