import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--dataset_type',type=str,default='imagenet100',
                        help="used in data_entry.py")
    parser.add_argument('--load_model_path',type=str,default="",
                        help="Model path for pretrain or test")
    parser.add_argument('--val_list',type=str,default='/root/autodl-tmp/imagenet100/val_list.txt',
                        help="Val list in training or test list in test")
    parser.add_argument('--label_dict',type=str,default='/root/autodl-tmp/imagenet100/label_dict.json',
                        help='label_dict.json')
    parser.add_argument('--seed',type=int,default=43,
                        help="manual random seed")
    return parser


def parse_train_args(parser):
    parser=parse_common_args(parser)
    # Some hyperparameters
    parser.add_argument('--lr',type=float,default=1e-1,
                        help="Learning rate")
    parser.add_argument('--momentum',type=float,default=0.9,
                        help="Momentum for SGD,alpha parameter for Adam")
    parser.add_argument('--weight_decay','--wd',type=float,default=1e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size',type=int,default=256,
                        help='Batch Size')
    parser.add_argument('--epochs',type=int,default=100,
                        help='Epochs to train the model')
    parser.add_argument('--lr_steps',type=int,default=30,
                        help='How many steps for lr_scheduler to update the lr')
    parser.add_argument('--lr_gamma',type=float,default=0.1,
                        help='Gamma param for lr scheduling')
    # Some settings
    parser.add_argument('--train_list',type=str,default='/root/autodl-tmp/imagenet100/train_list.txt',
                        help='Train list')
    parser.add_argument('--save_model_dir',type=str,default='/root/deepLearningCode/checkpoints/ResNet50',
                        help='Directory for saving checkpoints,logs and others')
    parser.add_argument('--model_saving_interval',type=int,default=10,
                        help='Interval for model saving')

    return parser



def parse_test_args(parser):
    parser=parse_common_args(parser)
    parser.add_argument('--save_result_dir',type=str,default='/root/deepLearningCode/checkpoints/ResNet50',
                        help='Directory for saving test logs,test results and so on')
    
    return parser


def get_train_args():
    parser=argparse.ArgumentParser()
    parser=parse_train_args(parser)
    args=parser.parse_args()
    return args


def get_test_args():
    parser=argparse.ArgumentParser()
    parser=parse_test_args(parser)
    args=parser.parse_args()
    return args
