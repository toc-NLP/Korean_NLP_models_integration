# @Author : juhyounglee
# @Datetime : 2020/08/01 
# @File : model_HCL_CLSTM_CLSTM.py
# @Last Modify Time : 2020/08/01
# @Contact : juhyounglee@{yonsei.ac.kr}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


"""
Neural Networks model : ensemble with Hierarchical C-SLTM, C-LSTM and C-LSTM 
"""
class CLSTM1(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CLSTM1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        D = args.embed_dim
        C = args.class_num
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        Ck = args.clstm_kernel
        V = args.embed_num
        D = args.embed_dim
        if(weight==None):
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

        #self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # pretrained  embedding
        #if args.word_Embedding:
        #    self.embed.weight.data.copy_(args.pretrained_weight)
            
        
        KK=[]
        for K in Ck:
            KK.append( K + 1 if K % 2 == 0 else K)

        self.convs_1d = nn.ModuleList([nn.Conv2d(1, F, (k, D), padding=(k//2,0)) for k in KK])
        
            
        self.bilstm = nn.LSTM(F, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # gru
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x

    def forward(self, input):
        input = self.embed(input)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)

        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
    
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

        logit = self.hidden2label(bilstm_out)
        
        return logit
    
class CLSTM2(nn.Module):
    
    def __init__(self, args, weight=None):
        super(CLSTM2, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        D = args.embed_dim
        C = args.class_num
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        Ck = args.clstm_add_kerneli
        V = args.embed_num
        D = args.embed_dim
        if(weight==None):
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()

            
        self.convs_1d = nn.ModuleList([nn.Conv2d(1, F, (K, D)) for K in Ck])  
        self.bilstm = nn.LSTM(F, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # gru
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x

    def forward(self, input):
        input = self.embed(input)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)

        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
    
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

        logit = self.hidden2label(bilstm_out)
        
        return logit
    
    
class wordCLSTM(nn.Module):
    def __init__(self, args, weight=None):

        super(wordCLSTM, self).__init__()
        
        #V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        self.num_layers = args.lstm_num_layers
        Wk = args.word_kernel

        V = args.embed_num
        D = args.embed_dim
        if(weight==None):
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()


        KK=[]
        for K in Wk:
            KK.append( K + 1 if K % 2 == 0 else K)

            
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(Ci, F, (k, D), padding=(k//2,0)) 
            for k in KK])
        
      
        self.bilstm = nn.LSTM(F, H, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        

        
        self.dropout = nn.Dropout(args.dropout)
        
      
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x

    def forward(self, input):
        input = self.embed(input)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
       
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        return bilstm_out.unsqueeze(0)

    
    
    
class sentCLSTM(nn.Module):

    def __init__(self, args):
        super(sentCLSTM, self).__init__()
       # V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        Sk = args.sent_kernel
        self.num_layers = args.lstm_num_layers


        KK=[]
        for K in Sk:
            KK.append( K + 1 if K % 2 == 0 else K)
            
        
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(Ci, F, (k, H*2), padding=(k//2,0)) 
            for k in KK])
        
        #self.convs_1d = [nn.Conv2d(Ci, F, (K, H*2), stride=1, padding=(K//2, 0)) for K in KK]

        self.bilstm = nn.LSTM(F, H, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden2label1 = nn.Linear(H*2, C)

    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x
    
    def forward(self, input):
        input = torch.transpose(input, 1,0)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = self.hidden2label1(bilstm_out)

        return bilstm_out
    
    
class HCL_CLSTM_CLSTM(nn.Module):
    
    def __init__(self, args,weight=None):
        super(HCL_CLSTM_CLSTM, self).__init__()
        
        V = args.embed_num
        D = args.embed_dim

       
        if(weight==None):
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = nn.Embedding.from_pretrained(weight).cuda()
        self.wordCLSTM = wordCLSTM(args,weight)
        self.senCLSTM = sentCLSTM(args)
        self.CLSTM1= CLSTM1(args,weight)
        self.CLSTM2 = CLSTM2(args,weight)
        

        self.relu = nn.ReLU()

    def forward(self, embed, source,max_sents):
        s = None
        for i in range(max_sents):
            _s = self.wordCLSTM(embed[:,i,:])
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)    

        logits_HCL = self.senCLSTM(s)
        logits_CLSTM1 = self.CLSTM1(source)
        logits_CLSTM2 = self.CLSTM2(source)
        
        logits = logits_HCL+logits_CLSTM1+logits_CLSTM2
     
        return F.log_softmax(logits, dim=1)
