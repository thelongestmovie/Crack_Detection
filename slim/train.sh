python train_image_classifier.py \
--train_dir=/home/yangyuhao/data/road/RUNS/2_RUNS/boost \
--dataset_name=road \
--dataset_split_name=train \
--dataset_dir=/home/yangyuhao/data/road/data/2_data/tf-records \
--batch_size=32 \
--max_number_of_steps=20000 \
--model_name=inception_v3 \
--save_interval_secs=600 \
--save_summaries_secs=60 \
--log_every_n_steps=50 \
--clone_on_cpu=none \
--dataset_num_samples=2995535 \
--checkpoint_path=/home/yangyuhao/data/road/RUNS/1_RUNS/boost


python train_image_classifier.py \
--train_dir=/home/yangyuhao/data/road/data/test_data/label_filter/RUN_50 \
--optimizer=adadelta  \
--dataset_name=road  \
--dataset_split_name=train \
--dataset_dir=/home/yangyuhao/data/road/data/test_data/label_filter/tf_data_50 \
--batch_size=32 \
--max_number_of_steps=2000000 \
--model_name=mobilenet_v2 \
--save_interval_secs=500 \
--save_summaries_secs=60 \
--log_every_n_steps=50 \
--dataset_num_samples=81240 \
--learning_rate=0.01 \
--num_epochs_per_decay=0.5 \
--weight_dacay=0.0 \
--cuda_visible_devices=1 \
--train_image_size=50


python train_image_classifier_bkp.py \
--train_dir=/home/yangyuhao/data/road/data/test_data/label_filter/RUN_test \
--optimizer=adam  \
--dataset_name=road  \
--dataset_split_name=train \
--dataset_dir=/home/yangyuhao/data/road/data/test_data/label_filter/tf_data_50 \
--batch_size=32 \
--max_number_of_steps=210000 \
--model_name=mobilenet_v2 \
--save_interval_secs=500 \
--save_summaries_secs=60 \
--log_every_n_steps=50 \
--dataset_num_samples=2995535 \
--learning_rate=0.01 \
--num_epochs_per_decay=0.5 \
--weight_dacay=0.0 \
--cuda_visible_devices=0,3 \
--train_image_size=50 \
--num_clones=2

50:2995535
100:13400