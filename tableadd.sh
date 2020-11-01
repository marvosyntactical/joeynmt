#!/bin/bash

strindex() { 
	  x="${1%%$2*}"
	  [[ "$x" = "$1" ]] && echo -1 || echo "${#x}"
}

reverse() {
	copy=$1
	len=${#1}
	for((i=$len-1;i>=0;i--)); do rev="$rev${copy:$i:1}"; done
	echo $rev
}


models_path="modelsadd/"
models=$(ls $models_path)
for model in $models; do
	if [ -e "$models_path$model/best.ckpt" ]
	then
		path=$models_path$model
		validstxt=$path/validations.txt

		model_short=${model:2}


		best_bleu=$(awk '{print $8}' $validstxt | sort -n | tail -n 1)
		best_ppl=$(awk '{print $6}' $validstxt | sort -n | tail -n 1)
		best_entF1=$(awk '{print $16}' $validstxt | sort -n | tail -n 1)
		best_entMCC=$(awk '{print $18}' $validstxt | sort -n | tail -n 1)

		echo "    " $model_short '&' ${best_bleu:0:5} '&' ${best_ppl:0:5} '&' ${best_entF1:0:5} '&' ${best_entMCC:0:5} '\\' 
	fi
done
