dir1="dstc2-restaurant/"
dir2="cluster_dstc2_data/"


files_dir1=$(ls $dir1)

for file in $files_dir1; do
	dir1file=$dir1$file
	dir2file=$dir2$file
	if [ -e $dir2file ]; 
	then
		diff $dir1file $dir2file;
	else
		echo "$file is in $dir1 but not $dir2...";
	fi
done
