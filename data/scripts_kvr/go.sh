#!/bin/bash

splitparts=( train dev test )
scriptsInOrder=(normalize_scenarios split_normalized_scenarios kbcanonize)

for script in "${scriptsInOrder[@]}"
do
	for i in "${splitparts[@]}"
	do
		if python3 ${script}.py $i; then
			echo "Successfully executed ${script}.py $i"
		else
			echo "Tried to execute ${script}, that didnt work."
			exit 1
		fi
	done
done

python3 canonize.py

cd ../kvr/

cat train.trvFINAL dev.trvFINAL test.trvFINAL > global.tmp
cat global.tmp | sort | uniq > global.trv
rm global.tmp

echo "all done letsgo"
