#! /bin/bash
python /root/deepLearningCode/models/ResNet/eval.py --load_model_path=./checkpoints/ResNet50/ResNet50_epoch_100.pth.tar --val_list=/root/autodl-tmp/imagenet100/test_list.txt