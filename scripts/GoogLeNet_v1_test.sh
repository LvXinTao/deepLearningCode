#! /bin/bash
python /root/deepLearningCode/models/GoogLeNet_v1/eval.py --load_model_path=./checkpoints/GoogLeNet_v1/GoogLeNet_v1_epoch_80.pth.tar --val_list=/root/autodl-tmp/imagenet100/test_list.txt