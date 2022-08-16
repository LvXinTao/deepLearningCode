import os
import argparse
import random
import json

parser=argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default="/root/autodl-tmp/imagenet100",
                    help="Imagenet100 dataset directory")
parser.add_argument('--train_PCT',type=float,default=0.8,
                    help='Train percent in imagenet100 dataset')
parser.add_argument('--val_PCT',type=float,default=0.1,
                    help="Validation percent in imagenet100 dataset")
parser.add_argument('--test_PCT',type=float,default=0.1,
                    help="Test percent in imagenet100 dataset")


if __name__=='__main__':
    args=parser.parse_args()

    label_dict_path=os.path.join(args.data_dir,'label_dict.json')
    train_list_txt=os.path.join(args.data_dir,"train_list.txt")
    val_list_txt=os.path.join(args.data_dir,"val_list.txt")
    test_list_txt=os.path.join(args.data_dir,"test_list.txt")

    train_list=list()
    val_list=list()
    test_list=list()
    
    # 分割出train,val,test
    for dir in os.listdir(args.data_dir):
        files=os.listdir(os.path.join(args.data_dir,dir))
        random.shuffle(files) # 随机打乱
        train_num=len(files)*args.train_PCT
        val_num=len(files)*args.val_PCT
        for i,file in enumerate(files):
            if i<train_num:
                output_list=train_list
            elif i<train_num+val_num:
                output_list=val_list
            else:
                output_list=test_list
            file_path=os.path.join(args.data_dir,dir,file)
            output_list.append(file_path+'\n')
    
    # 写入txt
    for txt,file_list in zip([train_list_txt,val_list_txt,test_list_txt],
                            [train_list,val_list,test_list]):
        with open(txt,'w') as f:
            f.writelines(file_list)

    # 将类别的dict写入json
    label_dict={k:v for v,k in enumerate([file for file in os.listdir(args.data_dir)
                                        if not file.endswith('.txt')])}
    with open(label_dict_path,'w') as f:
        json.dump(label_dict,f)
    