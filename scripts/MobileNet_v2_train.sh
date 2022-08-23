#! /bin/bash
python /root/deepLearningCode/models/MobileNet_v2/train.py --load_pretrain_path=models/MobileNet_v2/mobilenet_v2-pretrained.pth
shutdown -h now