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


models_path=$1
models=$(ls $models_path)
for model in $models; do
	if [ -e "$models_path$model/best.ckpt" ]
	then
		path=$models_path$model
		validstxt=$path/validations.txt

		gridcombo=${model:5}
		hops=${gridcombo:0:1}
		feeding_combo=${gridcombo:1:2}
		if [ ${feeding_combo:1:1} == "1" ]
		then
			feed="gated"
		else
			if [ ${feeding_combo:0:1} == "1" ]
			then
				feed="linear"
			else
				feed="none"
			fi
		fi
		dimensions=${gridcombo:4:1}
		if [ $dimensions == "1" ]
		then
			dims="2"
		else
			dims="1"
		fi
		reverse_combo=$(reverse $gridcombo)
		last_x_position=$(strindex $reverse_combo "x")
		positional_encoding="${reverse_combo:${last_x_position}-1:1}"
		if [ $positional_encoding == "1" ]
		then
			pos_enc="True"
		else
			echo $positional_encoding
			pos_enc="False"
		fi


		best_bleu=$(awk '{print $8}' $validstxt | sort -n | tail -n 1)
		best_ppl=$(awk '{print $6}' $validstxt | sort -n | tail -n 1)
		best_entF1=$(awk '{print $16}' $validstxt | sort -n | tail -n 1)
		best_entMCC=$(awk '{print $18}' $validstxt | sort -n | tail -n 1)

		echo "    " $hops '&' $feed '&' $dims '&' $pos_enc '&' ${best_bleu:0:5} '&' ${best_ppl:0:5} '&' ${best_entF1:0:5} '&' ${best_entMCC:0:5} '\\' 
	fi
done
	
