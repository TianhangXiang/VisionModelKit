CUDA_VISIBLE_DEVICES=7 python launch.py \
--data_dir /mnt/dataset/imagenet \
--gpu 0 \
--model resnet50 \
--dataset_name imagenet \
--b 256 \
--extra_note one_card_3090 \

CUDA_VISIBLE_DEVICES=7 python launch.py \
--data_dir /mnt/cephfs/mixed/dataset/imagenet \
--gpu 0 \
--model resnet50 \
--dataset_name imagenet \
--b 128 \
--extra_note one_card_xp \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ddp_launch.py ddp_train.py \
--data_dir /mnt/cephfs/mixed/dataset/imagenet \
--model resnet50 \
--dataset_name imagenet \
--b 128 \
--extra_note 6_card_xp \