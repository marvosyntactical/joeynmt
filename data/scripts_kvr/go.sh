#!/bin/bash

# how to generate differently preprocessed data for KVRN:
# Step 1. make desired change to appropriate script in scriptsInOrder
# Step 2. execute 'bash go.sh NEWFILEEXTENSION' with custom NEWFILEEXTENSION to avoid overwriting other preprocessed files
# Step 3. in the joeynmt config, change the data entry values to "valueNEWFILEEXTENSION"
# Step 4. Done


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

cd ../kvr/

cat train.trv$EXT dev.trv$EXT test.trv$EXT > global.tmp
cat global.tmp | sort | uniq > global.trv$EXT
rm global.tmp

echo "all done <3"
