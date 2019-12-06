epochs="200"


for mf in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do
	python start_ln2l.py \
		--dataset adult --task-mixing-factor $mf \
		--lr 1e-3 --batch-size 128 --weight-decay 0 --epochs $epochs \
        --entropy-weight 0.01 \
        --results-csv ln2l_adult_lr_disc
done

