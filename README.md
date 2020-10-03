[![Conference](http://img.shields.io/badge/ECCV-2020-4b44ce.svg)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710562.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2008.05248-b31b1b.svg)](https://arxiv.org/abs/2008.05248)


# Null-sampling for Invariant and Intepretable Representations

Implementation of our paper [Null-sampling for Invariant and Intepretable Representations](https://arxiv.org/abs/2008.05248).

## Requirements

Python 3.8 (or higher)

## Installing dependencies

The dependencies are listed in the `setup.py`.
To install them all, do

```
pip install -e .
```

## Running the code
Training of the CelebA cFlow model can be reproduced for CelebA and cMNIST, respectively,
with the folowing commands

```
start_inn.py --dataset celeba --levels 3 --level-depth 32 --glow True --reshape-method squeeze --autoencode False --input-noise True --quant-level 5 --use-wandb True --factor-splits 0=0.5 1=0.5 --train-on-recon False --recon-detach False --batch-size 32 --nll-weight 1 --pred-s-weight 1e-2 --zs-frac 0.001 --coupling-channels 512 --super-val True --super-val-freq 10 --val-freq 1 --task-mixing 0.5 --gpu 0 --num-discs 10 --disc-channels 512 --data-split-seed 42 --epochs 30
```

```
start_inn.py --dataset cmnist --levels 3 --level-depth 24 --glow True --reshape-method squeeze --autoencode False --input-noise True --quant-level 5 --use-wandb True --factor-splits 0=0.5 1=0.5 --train-on-recon False --recon-detach False --batch-size 256 --test-batch-size 512 --nll-weight 1 --pred-s-weight 1e-2 --zs-frac 0.002 --coupling-channels 512 --super-val True --super-val-freq 5 --val-freq 1 --task-mixing 0 --gpu 0 --num-discs 1 --disc-channels 512 --level-depth 24 --num-discs 3
```


