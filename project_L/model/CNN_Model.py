import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class  CNN_Text(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CNN_Text,self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if(weight==None):
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()
        
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)   # (N,W,D)

        if self.args.static:
            x = Variable(x.data)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        
        x = self.dropout(x)     # (N,len(Ks)*Co)
        logit = self.fc1(x)     # (N,C)
        return logit

