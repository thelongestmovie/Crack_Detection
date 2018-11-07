python test_image_classifier_single.py \
--checkpoint_path=/home/yangyuhao/data/road/data/test_data/label_filter/RUN \
--test_path=/home/yangyuhao/data/road/data/test_data/G314A-1029+358440-1029+526840.jpg/jpg_data \
--num_classes=2 \
--model_name=mobilenet_v2 \
--test_image_size=100

python test_image_classifier_batch_bkp.py \
--checkpoint_path=/home/yangyuhao/data/road/data/test_data/label_filter/RUN_50 \
--test_dir=/home/yangyuhao/data/road/data/test_data/G314A-1058+991800-1059+161600.jpg \
--batch_size=32 \
--num_classes=2 \
--model_name=mobilenet_v2 \
--test_image_size=50 \
--cuda_visible_devices=0


python test_image_classifier_trt.py \
--checkpoint_path=/home/yangyuhao/data/road/data/test_data/label_filter/RUN_50 \
--test_dir=/home/yangyuhao/data/road/data/test_data/G314A-1077+917040-1078+090360.jpg \
--batch_size=32 \
--num_classes=2 \
--model_name=mobilenet_v2 \
--test_image_size=50 \
--cuda_visible_devices=0