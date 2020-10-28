models_dir="modelsadd/"
model_folders=$(ls $models_dir)
tb_plots="tensorboard_plots/"
tb="/tensorboard/"

for model_path in $model_folders; do
	tb_path=$tb_plots$model_path
	mkdir $tb_path 
	event_file_as_array=$(ls $models_dir$model_path$tb)
	for event_file in $event_file_as_array; do
		cp -rp $models_dir$model_path$tb$event_file $tb_path"/"
	done
done


