''' Import '''
#region
import sys,os
import joblib
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from itertools import chain
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import sys,os
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import datetime 
from multiprocessing import pool
import multiprocessing as mp
import torch
from torch import matmul, diagonal
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.cluster import KMeans
from functools import partial
import toml
from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import torch
import pickle
import time
from collections import OrderedDict
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from ast import literal_eval
from joblib import Parallel,delayed
import dill
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.utils.data import Dataset
import pytorch_lightning as pl
#endregion

''' Model initialize. '''
#region
state_dict = "/home/user/data3/wangjx/prj_007_6mA/myModel/new_240304/train_dataset/New_240314_bbasic/Mouse/new_4318_whole_freq0.5/train_results/kfold2_model.0521.2010.pt"
sys.path.append("/home/user/data3/wangjx/prj_007_6mA/Arrange_allData_240529/Step2ModelTrainingAndInference/model_training_240603/model")
from model_ccs_attn import model
model = model()
model.load_state_dict(torch.load(state_dict))
#endregion

''' Read args. '''
#region
def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=True
    )
    parser.add_argument("--fname", default=None, required=True)
    parser.add_argument("--num_workers", default=10, type=int,
    help='Number of workers to use while running.')
    parser.add_argument("--output", default=None,required=True)
    return parser

args = argparser().parse_args()

''' File readin. '''
#region
fname = args.fname
sys.path.append("/home/user/data3/wangjx/prj_007_6mA/Arrange_allData_240529/Step2ModelTrainingAndInference/scripts/inference_for_pred_240529/RawIPDandPW_21mer")

mm = []

import gc
gc.disable()

with open(fname,'rb') as f:
    try:
        while True:
            mm.append(dill.load(f))
    except EOFError:
            pass

gc.enable()
f.close()

print("File readin done.")

#endregion

''' Data processing. '''
loc = {'A':0,
          'G':1,
          'T':2,
          'C':3}

def encoder(seq,ipdList):
    encd = np.zeros((4,len(ipdList)))
    new_seq = []
    for idx,i in enumerate(seq):
        new = i
        loca = loc[new]
        ipd = ipdList[idx]
        encd[loca,idx] = ipd
    return np.array(encd.tolist())

def minUmax(array):
    norm = lambda x: (x - min(x))/(max(x) - min(x))
    return np.apply_along_axis(norm,1,array)

class torchDataset_inference(Dataset):

    def __init__(self,X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.X[idx][0])
        return feature
    
batch_size = 2048

def random_fn(x):
    np.random.seed(datetime.datetime.now().second)

def inference(model,dl,device):
    """
    Run inference on unlabelled dataset
    """
    model.eval()
    all_y_pred = []
    with torch.no_grad():
        kmers = []
        for batch in dl:
            y_pred = model(batch)
            y_pred = y_pred.detach().cpu().numpy()
            all_y_pred.extend(y_pred.flatten())
    return all_y_pred

''' Remove PW info. '''
def Run_epoch(i):
    ipdList = []
    if '-' in  i.kmerList or \
    np.isnan(i.ipdList).any() == True: 
        return None,None
    else:
        ipd_list = minUmax(i.ipdList)
        for j in ipd_list:
            ipdList.append(encoder(i.kmerList,j))
        ipdList = np.array(ipdList)
        merge = ipdList
        return merge,i.ID
        
def prep(pickle,num_workers = 10):
    result = Parallel(n_jobs = num_workers)(delayed(Run_epoch)(i) for i in pickle)

    mergedArray = [(x,y) for x,y in result if x is not None and y is not None]

    IDs = [x[1] for x in mergedArray]

    X = mergedArray
    y = IDs

    inference_ds = torchDataset_inference(X)
    inference_dl = DataLoader(inference_ds,
                batch_size=batch_size, worker_init_fn=random_fn,
                shuffle=False)
    
    return inference_dl,IDs

inference_dl,IDs = prep(mm)

del mm 

print("Data processing done.")

y_pred = inference(model,inference_dl,"cpu")

print("Model inference done.")

result = pd.DataFrame({'ID':IDs,'y_pred':y_pred})
result.to_csv(args.output,sep = "\t",index = None)

print("All inference done.")



