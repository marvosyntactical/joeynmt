#!/bin/bash

#TODO add more options like par

#signature:
name=$1
#duration should have syntax 2-12:15:59
duration=${2:-"3-00:00:00"}
memory=${3:-"128000"}
partition=${4:-"students"}

clear

echo "Starting procedure for job $name ..."
echo


#paths
cfg_path=configs/kvr/
sbatch_path=sbatch/
model_path=models/



#extensions
cfg_ext=".yaml"
sbatch_ext=".sh"


#================================================
#procedure :

# find config
config="$cfg_path$name$cfg_ext"



if [ -e "$config" ]
then
	echo "Found $config, resuming ..."

else
	echo "$name does not exist. Please create this config file: $cfg_path$name$cfg_ext, then execute this script again ;)"
	exit 1
fi


# create model dir:

# find prefix numer: highest existing +1
model_folder=$(ls $model_path)
numbered=()
for model in $model_folder; do
	prefix=(${model//_/ })
	numbered+=($prefix)
done

max=${numbered[0]}
for n in "${numbered[@]}" ; do
	# what is this magic
	case $max in
		''|*[!0-9]*) echo "Found some ${model_path}/folder starting with something other than a number ..." ;;
		*) ((n > max)) && max=$n ;;
	esac
done

let new_prefix=$max+1
date=$(date +"%d-%m")

model_dir="${new_prefix}_${name}_$date"


# if model dir with name exists, fail

if [ -d $model_dir ] ; then
	echo "$model_folder already exists, exiting."
	exit 1
fi

# in config, replace model_dir: blabla with model_dir: $model_path$model_dir

model_path_no_slash=${model_path//\//}
sed -i "s/model_dir: [^#]*#/model_dir: \"${model_path_no_slash}\/${model_dir}\" #/g" $config

sbatch="$sbatch_path$name$sbatch_ext"
echo "creating sbatch $sbatch:"

template="${sbatch_path}template$sbatch_ext"
cp -rp "$template" "$sbatch"

#replace all occurences of JOBNAME with name
sed -i "s/JOBNAME/$name/g" $sbatch

#more replacements
sed -i "s/DURATION/$duration/g" $sbatch
sed -i "s/MEMORY/$memory/g" $sbatch
sed -i "s/PARTITION/$partition/g" $sbatch

echo
echo "---------------------------------------------------------------------------"
cat $sbatch
echo "---------------------------------------------------------------------------"
echo 
echo "Should we execute this? [Y/n]"
read input

if [ $input = n ]; then
	echo "Alright, never mind then. :("
	exit 1
fi 

# execute sbatch:
sbatch $sbatch
echo
squeue
echo
echo model can be found under $model_path$model_dir

echo 
echo "==========================================================================="
echo 
echo "    starting ${new_prefix}th training job named $name, now go away and pray"
echo
echo "                                    🛐"
echo
echo "==========================================================================="
echo





