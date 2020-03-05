

for mix_fact in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do
python start_vae.py --kl-weight 1 --super-val True --val-freq 10 --epochs 0 \
--dataset celeba --levels 5 --pred-s-weight 1e-2 --recon-loss l1 --elbo-weight 1e-2 \
--eval-epochs 40 --results-csv stellar_lioncvae_celeba_results_Gender_Smiling --gpu 0 --task-mixing-factor $mix_fact \
 $@
done
done
