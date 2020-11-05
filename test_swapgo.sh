#!/bin/bash
source ~/.bashrc	


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


# call this script e.g. like so:
# $ ./testgo.sh nameOfLatestConfig modelsadd/ 0-06:00:00 64000 gpushort 

# script call signature:

name=$1
model_path=${2:-"models/"}
duration=${3:-"0-16:00:00"} # duration should have syntax 2-12:15:59
memory=${4:-"128000"}
partition=${5:-"gpulong"}


clear

echo "Starting procedure for job $name ..."
echo


#paths
cfg_path=configs/kvr/swapped_testruns/
sbatch_path=sbatch/

#extensions
cfg_ext=".yaml"
sbatch_ext=".sh"


#================================================
#procedure :

# find config
config="$cfg_path$name$cfg_ext"

# best checkpoint has to be provided under config.load_model
ckpt_path=$(grep "load_model" $config | awk '{print $2}')
echo $ckpt_path

if [ -e "$config" ]
then
	echo "Found $config, resuming ..."

else
	echo "$name does not exist. Please create this config file: $config, then execute this script again ;)"
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
		''|*[!0-9]*) echo "Warning: Found some ${model_path}/folder starting with something other than a number ..." ;;
		*) ((n > max)) && max=$n ;;
	esac
done

let new_prefix=$max+1
date=$(date +"%d-%m")

model_dir="${new_prefix}_${name}_$date"


# if model dir with name exists, fail

if [ -d $model_dir ] ; then
	echo "$model_folder already exists, exiting. Make sure youre using a new config name that hasnt been used in a previous training run ... today."
	exit 1
fi

# in config, replace model_dir: blabla with model_dir: $model_path$model_dir

model_path_no_slash=${model_path//\//}
sed -i "s/model_dir: [^#]*#/model_dir: \"${model_path_no_slash}\/${model_dir}\" #/g" $config

sbatch="$sbatch_path$name$sbatch_ext"
echo "creating sbatch $sbatch:"

template="${sbatch_path}testtemplate$sbatch_ext"
cp -rp "$template" "$sbatch"

#replace all occurences of JOBNAME with name
sed -i "s/JOBNAME/$name/g" $sbatch


sed -i "s/BRUH/configs\/kvr\/testruns\//g" $sbatch

#more replacements
sed -i "s/DURATION/$duration/g" $sbatch
sed -i "s/MEMORY/$memory/g" $sbatch
sed -i "s/PARTITION/$partition/g" $sbatch
ckpt_path_escape_slash=${ckpt_path//\//\\/}
echo $ckpt_path_escape_slash
sed -i "s/LOADMODEL/$ckpt_path_escape_slash/g" $sbatch

echo
echo "---------------------------------------------------------------------------"
cat $sbatch
echo "---------------------------------------------------------------------------"
echo 

loadmodel_not_replaced=$(grep LOADMODEL $sbatch | wc -l)
if [ $loadmodel_not_replaced == "0" ]
then
	echo "successfully replaced LOADMODEL"
else
	echo "failed to replace LOADMODEL above for some reason"
	exit
fi


echo $ckpt_path_escape_slash


echo
echo "---------------------------------------------------------------------------"
cat $config
echo "---------------------------------------------------------------------------"
echo 

# execute sbatch:
jobnum=$(sbatch $sbatch | awk 'NF>1{print $NF}')
jobinfo=$(squeue | grep $jobnum)

echo
echo "Started Job number $jobnum"
echo
echo "$jobinfo"
echo
echo model can be found under $model_path$model_dir

echo 
echo "==========================================================================="
echo 
echo "    starting ${new_prefix}th training job named $name, now go away and pray"
echo
echo "                                   🛐"
echo
echo "==========================================================================="
squeue | grep 'koss'
echo





