from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image



class imagenet100Dataset(Dataset):
    def __init__(self,args,is_train,transform=None):
        super(imagenet100Dataset,self).__init__()
        if is_train:
            data_list=args.train_list
        else:
            data_list=args.val_list
        self.transform=transform
        with open(data_list,'r') as f:
            self.img_paths=f.read().splitlines()
        with open(args.label_dict,'r') as f:
            self.label_dict=json.load(f)
    

    def __len__(self):
        return len(self.img_paths)

    
    def __getitem__(self,idx):
        image=Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            image=self.transform(image)
        label=self.label_dict[self.img_paths[idx].split('/')[-2]]
        return image,label


