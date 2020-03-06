for mf in "0.0" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "0.95" "1.0"
do
	python -u start_ln2l.py \
	--dataset celeba --task-mixing-factor $mf \
	--lr 1e-3 --batch-size 128 --weight-decay 0 --epochs 25 \
        --entropy-weight 0.01 \
        --results-csv ln2l_celeba.csv "$@"
done

echo "Finished job script"

