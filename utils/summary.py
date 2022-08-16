from tensorboardX import SummaryWriter
import os
import torch


class Recorder():
    def __init__(self):
        self.metrics={}

    def record(self,name,value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name]=[value]
    
    def conclude(self):
        # 每个epoch计算一次平均值
        kvs={}
        for key in self.metrics.keys():
            kvs[key]=sum(self.metrics[key])/len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key]=[]
        return kvs


class Summary:
    def __init__(self,args):
        self.writer=SummaryWriter('/root/tf-logs')
        self.recorder=Recorder()

    def record_scalar(self,name,value):
        self.recorder.record(name,value)

    def save_curves(self,epoch):
        kvs=self.recorder.conclude()
        for key in kvs.keys():
            self.writer.add_scalar(key,kvs[key],epoch)
        return kvs