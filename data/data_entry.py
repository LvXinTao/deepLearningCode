from data.imagenet100_dataset import imagenet100Dataset
import data.imagenet100_augment as imagenet100_augment

from torch.utils.data import DataLoader


def get_dataset_by_type(args):
    type2data={
        'imagenet100':imagenet100Dataset
    }
    dataset=type2data[args.dataset_type]
    return dataset


def get_num_features_by_type(args):
    type2num={
        'imagenet100':100
    }
    return type2num[args.dataset_type]


def fetch_train_loader(args):
    train_dataset=get_dataset_by_type(args)(args,True,imagenet100_augment.train_transformer())
    train_loader=DataLoader(train_dataset,args.batch_size,shuffle=True,
                            num_workers=4,pin_memory=True,drop_last=False)
    return train_loader


def fetch_eval_loader(args):
    eval_dataset=get_dataset_by_type(args)(args,False,imagenet100_augment.eval_transformer())
    val_loader=DataLoader(eval_dataset,1,shuffle=False,
                            num_workers=1,pin_memory=True,drop_last=False)
    return val_loader



