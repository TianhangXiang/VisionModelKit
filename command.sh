CUDA_VISIBLE_DEVICES=7 python launch.py \
--data_dir /mnt/dataset/imagenet \
--gpu 0 \
--model resnet50 \
--dataset_name imagenet \
--b 256 \
--extra_note one_card_3090 \