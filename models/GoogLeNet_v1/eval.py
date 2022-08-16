import sys
sys.path.append('/root/deepLearningCode')

import warnings
warnings.filterwarnings('ignore')

from data.data_entry import fetch_eval_loader,get_num_features_by_type
from models.GoogLeNet_v1.model import GoogLeNet_v1
from models.GoogLeNet_v1.options import get_test_args
from utils.logger import set_logger
from utils.config import set_seed
from utils.summary import Summary

import os
import logging
import torch
import torch.nn
from tqdm import tqdm
import numpy as np


class Evaluator:
    def __init__(self):
        self.args=get_test_args()
        if not os.path.exists(self.args.save_result_dir):
            os.system('mkdir -p '+self.args.save_result_dir)
        set_seed(self.args.seed)
        set_logger(os.path.join(self.args.save_result_dir,'test.log'))
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.summary=Summary(self.args)
        self.test_loader=fetch_eval_loader(self.args)

        self.model=GoogLeNet_v1(get_num_features_by_type(self.args),
                                aux_logits=False,
                                init_weights=False)
        if self.args.load_model_path=='':
            raise Exception("No model loaded!")
        elif not os.path.exists(self.args.load_model_path):
            raise Exception("No model found at {}".foramt(self.args.load_model_path))
        checkpoint=torch.load(self.args.load_model_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model'],strict=False) # 将aux的参数丢弃
        logging.info('----Model {} loaded!----'.format(self.args.load_model_path))
        self.model.to(self.device)

    def eval(self):
        logging.info('----Start testing!----')
        self.model.eval()
        correct=0.0
        total=len(self.test_loader)
        with torch.no_grad():
            with tqdm(total=total) as t:
                for test_batch,labels_batch in self.test_loader:
                    test_batch,labels_batch=test_batch.to(self.device),labels_batch.to(self.device)
                    output=self.model(test_batch)
                    
                    output=output.detach_().cpu().numpy()
                    labels_batch=labels_batch.detach_().cpu().numpy()
                    correct+=np.sum(np.argmax(output,axis=1)==labels_batch)
                    t.update()
        accuracy=correct/float(total)
        logging.info('----Accuracy:{}/{}={}%----'.format(correct,total,accuracy*100))


if __name__=='__main__':
    evaluator=Evaluator()
    evaluator.eval()
