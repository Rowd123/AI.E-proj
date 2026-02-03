import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import numpy as np 
import os 
import argparse 
from tqdm import tqdm, trange 
import random 
import pandas as pd 
from pathlib import Path 
import shutil 
import json 
from datasets import __dict__ as data_dict
from architectures import __dict__ as model_dict



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--base_config', type=str, required=True, help='config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.base_config is not None 
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg 

def copy_config(args, exp_root):
    p = Path(".")
    python_files = list(p.glob('**/*.py'))
    filtered_list = [
        file for file in python_files
        if ('checkpoints' not in str(file)
            and 'archive' not in str(file)
            and 'results' not in str(file))
        ]
    for file in filtered_list:
        file_dest = exp_root / 'src_code' / file 
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, file_dest)
        
    with open(exp_root / 'config.json', 'w') as fp:
        json.dump(args, fp, indent=4)


def hash_config(args: argparse.Namespace):
    res = 0 
    for i, (key,value) in enumerate(args.items()):
        if key != "port":
            if type(value)==str:
                hash_ = sum(value.encode())
            elif type(value) in [float, int, bool]:
                hash_ = round(value, 3)
            else:
                hash_ = sum([int(v) if type(v) in [float, int, bool] else sum(v.encode()) for v in value])
            res += hash_ * random.randint(1, 1000000)
    return str(res)[-10:].split('.')[0]

def get_loaders(args, data):
    size = {'MNIST': 28, 'MNISTM': 28, 'CIFAR10': 32, 'USPS': 28, 'SVHN': 28, 'STL10': 32}
    train_transforms = transforms.Compose([transforms.Resize(size[data]),
                                           transforms.CenterCrop(size[data]),
                                           transforms.ToTensor()
                                           ])
    
    test_transforms = transforms.Compose([transforms.Resize(size[data]),
                                         transforms.CenterCrop(size[data]),
                                         transforms.ToTensor()
                                         ])
    
    train_dataset = data_dict[data] = 
    

 

    
        
    