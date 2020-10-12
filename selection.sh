#!/bin/bash

models_path="modelsgrid/"
models=$(ls $models_path)
for model in $models; do
	if [ -e "$models_path$model/best.ckpt" ]
	then
		path=$models_path$model
		best_step=$(ls -l $path/best.ckpt | grep -o "\w*" | tail -n 2 | head -n 1)
		# echo $model "best:" $best_step
		validstxt=$path/validations.txt
		best_bleu=$(grep "Steps: $best_step\s" $validstxt | awk '{print $8}')
		echo $best_bleu $model $best_step
	fi
done
	
