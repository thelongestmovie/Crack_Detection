python eval_image_classifier.py \
--checkpoint_path=/home/yangyuhao/data/road/RUNS/3_RUNS \
--eval_dir=/home/yangyuhao/data/road/log \
--dataset_name=road \
--dataset_split_name=validation \
--dataset_num_samples=3296 \
--dataset_dir=/home/yangyuhao/data/road/data/validation_data/tf-records \
--model_name=inception_v3
