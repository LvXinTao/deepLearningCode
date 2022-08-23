import sys
# 去整个项目文件夹寻找模块
sys.path.append('/root/deepLearningCode')

import warnings
warnings.filterwarnings('ignore')

from data.data_entry import fetch_train_loader,fetch_eval_loader,get_num_features_by_type
from models.MobileNet_v2.model import MobileNetV2
from models.MobileNet_v2.options import get_train_args
from utils.logger import set_logger
from utils.config import set_seed
from utils.summary import Summary

import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self):
        self.args=get_train_args()
        # 创建保存模型的文件夹
        if not os.path.exists(self.args.save_model_dir):
            os.system('mkdir -p '+self.args.save_model_dir)
        # 设置随机种子和logger
        set_seed(self.args.seed)
        set_logger(os.path.join(self.args.save_model_dir,'train.log'))
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        # 设置Tensorboard
        self.summary=Summary(self.args)
        # 设置Dataloader
        self.train_loader=fetch_train_loader(self.args)
        self.val_loader=fetch_eval_loader(self.args)
        # 设置model
        self.model=MobileNetV2(get_num_features_by_type(self.args))
        # 如果有预训练模型，则导入,并进行微调
        if self.args.load_pretrain_path !='':
            assert os.path.exists(self.args.load_pretrain_path),"No model at {}!".format(self.args.load_pretrain_path)
            Pre_weights=torch.load(self.args.load_pretrain_path)
            # 仅导入卷积层参数
            Pre_dict={k:v for k,v in Pre_weights.items() if 'features' in k}
            self.model.load_state_dict(Pre_dict,strict=False)
            # 冻结模型feature层参数
            for param in self.model.features.parameters():
                param.requires_grad=False
            logging.info('----Pretrain model Loaded!----')
        self.model.to(self.device)
        # 设置optimizer
        self.optimizer=torch.optim.SGD([param for param in self.model.parameters() if
                                        param.requires_grad],self.args.lr,
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        # 设置scheduler
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.args.lr_steps,
                                                        gamma=self.args.lr_gamma)
        # 设置loss_fn
        self.loss_fn=nn.CrossEntropyLoss() # 调用__call__
        # 设置epochs
        self.epochs=0
        # 如果有断点，则导入
        if self.args.load_checkpoint_path !='':
            self.load_checkpoint()

    def train(self):
        logging.info('----Start training!----')
        for epoch in range(self.epochs,self.args.epochs):
            logging.info('----Epoch {}/{}----'.format(epoch+1,self.args.epochs))
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            if (epoch+1)%self.args.model_saving_interval==0:
                self.save_checkpoint(epoch+1)


    def train_per_epoch(self,epoch):
        self.model.train()        
        correct=0.0
        total=0.0
        with tqdm(total=len(self.train_loader)) as t:
            for i,(train_batch,labels_batch) in enumerate(self.train_loader):
                train_batch,labels_batch=train_batch.to(self.device),labels_batch.to(self.device)
                output=self.model(train_batch)

                loss=self.compute_loss(output,labels_batch)
                self.summary.record_scalar('loss',loss.item())

                pred=output.argmax(axis=1)
                correct+=torch.eq(pred,labels_batch).sum().float().item()
                total+=1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                t.update()
        self.scheduler.step()
        self.summary.record_scalar('accuracy',correct/total)
        kvs=self.summary.save_curves(epoch+1)
        logging.info('----Loss:{} ----'.format(kvs['loss']))
        logging.info('----Accuracy:{} ----'.format(kvs['accuracy']))
    
    
    def val_per_epoch(self,epoch):
        pass

    def compute_loss(self,output,labels_batch):
        loss=self.loss_fn(output,labels_batch)
        return loss
    
    def save_checkpoint(self,epoch):
        checkpoint={
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict() if self.scheduler else None,
            'epoch':epoch
        }
        torch.save(checkpoint,os.path.join(self.args.save_model_dir,'MobileNet_v2_epoch_{}.pth.tar'.format(epoch)))
        logging.info('----Checkpoint saved at epoch {}----'.format(epoch))

    
    def load_checkpoint(self):
        if not os.path.exists(self.args.load_model_path):
            raise Exception("No Checkpoint found at {}".format(self.args.load_model_path))
        checkpoint=torch.load(self.args.load_model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epochs=checkpoint['epoch']
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info('----Checkpoint {} loaded----'.format(self.args.load_model_path))


if __name__=="__main__":
    trainer=Trainer()
    trainer.train()