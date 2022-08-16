import sys
# 去整个项目文件夹寻找模块
sys.path.append('/root/deepLearningCode')

import warnings
warnings.filterwarnings('ignore')

from data.data_entry import fetch_train_loader,fetch_eval_loader,get_num_features_by_type
from models.GoogLeNet_v1.model import GoogLeNet_v1
from models.GoogLeNet_v1.options import get_train_args
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
        self.model=GoogLeNet_v1(get_num_features_by_type(self.args),
                                aux_logits=True,
                                init_weights=True)
        self.model.to(self.device)
        # 设置optimizer
        self.optimizer=torch.optim.SGD(self.model.parameters(),self.args.lr,
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        # 设置scheduler
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.args.lr_steps,
                                                        gamma=self.args.lr_gamma)
        # 设置loss_fn
        self.loss_fn=nn.CrossEntropyLoss() # 调用__call__
        # 如果有预训练模型或断点，则导入
        if self.args.load_model_path !='':
            self.load_checkpoint()

    def train(self):
        logging.info('----Start training!----')
        for epoch in range(self.args.epochs):
            logging.info('----Epoch {}/{}----'.format(epoch+1,self.args.epochs))
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            if (epoch+1)%self.args.model_saving_interval==0:
                self.save_checkpoint(epoch+1)


    def train_per_epoch(self,epoch):
        self.model.train()
        with tqdm(total=len(self.train_loader)) as t:
            for i,(train_batch,labels_batch) in enumerate(self.train_loader):
                train_batch,labels_batch=train_batch.to(self.device),labels_batch.to(self.device)
                output,aux_out2,aux_out1=self.model(train_batch)

                loss=self.compute_loss(output,aux_out2,aux_out1,labels_batch)
                accuracy=self.compute_accuracy(output,labels_batch)
                self.summary.record_scalar('loss',loss.item())
                self.summary.record_scalar('accuracy',accuracy)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                t.update()
        self.scheduler.step()
        kvs=self.summary.save_curves(epoch+1)
        logging.info('----Loss:{} ----'.format(kvs['loss']))
        logging.info('----Accuracy:{} ----'.format(kvs['accuracy']))
    
    
    def val_per_epoch(self,epoch):
        pass

    def compute_loss(self,output,aux_out2,aux_out1,labels_batch):
        loss_1=self.loss_fn(output,labels_batch)
        loss_2=self.loss_fn(aux_out2,labels_batch)
        loss_3=self.loss_fn(aux_out1,labels_batch)
        loss=loss_1+0.3*loss_2+0.3*loss_3
        return loss
    
    def compute_accuracy(self,output,labels):
        output=np.argmax(output.detach().cpu().numpy(),axis=1)
        return np.sum(output==labels)/float(labels.size()[0])

    def save_checkpoint(self,epoch):
        checkpoint={
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict() if self.scheduler else None,
            'epoch':epoch
        }
        torch.save(checkpoint,os.path.join(self.args.save_model_dir,'GoogLeNet_v1_epoch_{}.pth.tar'.format(epoch)))
        logging.info('----Checkpoint saved at epoch {}----'.format(epoch))

    
    def load_checkpoint(self):
        if not os.path.exists(self.args.load_model_path):
            raise Exception("No model found at {}".format(self.args.load_model_path))
        checkpoint=torch.load(self.args.load_model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info('----Checkpoint {} loaded----'.format(self.args.load_model_path))


if __name__=="__main__":
    trainer=Trainer()
    trainer.train()