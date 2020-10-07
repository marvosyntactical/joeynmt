#!/bin/bash

EXT=$1

scriptsInOrder=(parse_kvr_json normalize_scenarios split_normalized_scenarios kbcanonize canonize)
splitparts=( train dev "test" )

for script in "${scriptsInOrder[@]}"
do
	for i in "${splitparts[@]}"
	do
		if python3 ${script}.py $i $EXT; then
			echo "Successfully executed ${script}.py $i"
		else
			echo "Tried to execute ${script}, that didnt work."
			exit 1
		fi
	done
done

python3 canonize.py

cd ../kvr/

cat train.trv$EXT dev.trv$EXT test.trv$EXT > global.tmp
cat global.tmp | sort | uniq > global.trv$EXT
rm global.tmp

echo "all done <3"
