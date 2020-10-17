#!/bin/bash

models_path="modelsgrid/"
models=$(ls $models_path)
for model in $models; do
	if [ -e "$models_path$model/best.ckpt" ]
	then
		path=$models_path$model
		validstxt=$path/validations.txt
		best_bleu_n_step=$(awk '{print $8,$2}' $validstxt | sort -n | tail -n 1)
		echo $best_bleu_n_step $model
	fi
done
	
