''' Import packages '''
#region
import pickle
import pandas as pd
from recordtype import recordtype
from collections import OrderedDict
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from itertools import chain
from collections import defaultdict
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import sys,os
import re
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate
import datetime 
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
import time
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
#endregion

''' Model identification '''
#region
def get_activation(activation):
    activation_func = None
    if activation == 'tanh':
        activation_func = nn.Tanh()
    elif activation == 'sigmoid':
        activation_func = nn.Sigmoid()
    elif activation == 'relu':
        activation_func = nn.ReLU()
    elif activation == 'softmax':
        activation_func = nn.Softmax(dim=1)
    else:
        raise ValueError("Invalid activation")
    return activation_func

class SigmoidProdPooling(nn.Module):

    def __init__(self, input_channel, 
                 n_reads_per_site=5):
        super(SigmoidProdPooling, self).__init__()
        self.input_channel = input_channel
        self.n_reads_per_site = n_reads_per_site
        self.probability_layer = nn.Sequential(*[nn.Linear(input_channel, 1), get_activation('sigmoid')])

    def set_num_reads(self, n_reads_per_site):
        self.n_reads_per_site = n_reads_per_site
        
    def predict_read_level_prob(self, x):
        return self.probability_layer(x)
    
    def forward(self, x):
        read_level_prob = self.predict_read_level_prob(x)
        return 1 - torch.prod(1 - read_level_prob, axis=1)

class ccslevel_attn(pl.LightningModule):
    def __init__(self,pooling_filter):
        super(ccslevel_attn, self).__init__()
        self.pooling_filter = pooling_filter
        self.conv = nn.Sequential(
            # layer1
            nn.LazyConv2d(16, kernel_size=3, stride = 1,padding = 1),
            nn.LazyBatchNorm2d(),
            nn.SiLU(),
#             nn.AvgPool2d(2,2),
            
            # layer2
            nn.LazyConv2d(64, kernel_size=3, stride = 1,padding = 1),
            nn.LazyBatchNorm2d(),
            nn.SiLU(),
            
            # layer3
            nn.LazyConv2d(128, kernel_size=3, stride = 1,padding=1),
            nn.LazyBatchNorm2d()
            )  
        
        self.attn_conv = nn.LazyConv2d(128, kernel_size = 1, stride=1)
        
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Flatten(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(32),
            nn.Sigmoid()
            )      
        
    def forward_once(self,x):
        x = self.conv(x)
        maxp_x =  torch.mean(x,dim = 1,keepdim = True)
        avgp_x,_ = torch.max(x,dim = 1, keepdim = True)
        spat_attn = self.attn_conv(torch.cat([maxp_x, avgp_x], dim=1))
        new_x = torch.mul(spat_attn, x)
        feat = self.fc(new_x)
        return feat
    
    def get_subreads_probability(self, x):
        subreads_representation = x
        return self.pooling_filter(subreads_representation)
     
    def forward(self, img):
        # img = img['X']
        x1,x2,x3,x4,x5 = torch.split(img,1,dim = 1)
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        x3 = self.forward_once(x3)
        x4 = self.forward_once(x4)
        x5 = self.forward_once(x5)
        feature_concat = torch.stack([x1,x2,x3,x4,x5],axis = 1)
        output = self.get_subreads_probability(feature_concat)
        return output
    
ALPHA = 0.25
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

def model():
    pooling_filter = SigmoidProdPooling(32)
    model = ccslevel_attn(pooling_filter)
    return model

#endregion