from gensim import models
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.CNN_Model import CNN_Text
from model.HCL_2CLSTM_Model import HCL_CLSTM_CLSTM



import argparse
import os
import sys
import datetime
import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed_dim', type=int, default=100, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=list, default=[3], help='number of each kind of kernel')
parser.add_argument('-word_kernel', type=list, default=[3], help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-clstm_filter_size', type=str, default='3', help='clstm-filter')
parser.add_argument('-clstm_hidden_size',type=int,  default=300, help='fix the embedding')
parser.add_argument('-clstm_num_layers',type=int,  default=300, help='fix the embedding')
parser.add_argument('-lstm_num_layers',type=int,  default=300, help='fix the embedding')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()



embed_lookup = models.fasttext.load_facebook_model("cc.ko.300.bin")

word2index={}
for i in range(0, len(embed_lookup.wv.index2word)):
        try:
            word2index[embed_lookup.wv.index2word[i]] = i
        except KeyError:
            word2index[embed_lookup.wv.index2word[i]] = i




def tokenizer(inputText):
    reviews_words = inputText.split(' ')
    tokenized_reviews = []
    sentences = []
    for word in reviews_words:
        if('.' in word  or '?' in word  or '!' in word ):
            word = word.replace('.', '').replace('!','').replace('?','')
            try:
                sentences.append(word2index[word])
            except:
                sentences.append(0)
            if(len(sentences) >= 2):
                tokenized_reviews.append(sentences)
               # print("token##:", tokenized_reviews)
            sentences = []
            continue
        try:
            idx = word2index[word]
        except: 
            idx = 0
        sentences.append(idx)
    if(len(sentences)>=2):
        tokenized_reviews.append(sentences)
    return tokenized_reviews


print("Reading Train set......")


fr = open('./nsmc/ratings_train.tsv', 'r', encoding='utf-8')
lines = csv.reader(fr,  delimiter='\t')
X_train=[]
y_train=[]
meanSen = []
meanLine = []
for line in lines:
    if('label' in line[0]):
        continue
    w2i = list(tokenizer(line[1]))
    for subsentence in w2i:
        meanSen.append(len(subsentence))
    meanLine.append(len(w2i))
    X_train.append(w2i)
    y_train.append([int(line[0])])
fr.close()

print("Reading Test set......")

fr = open('./nsmc/ratings_train.tsv', 'r', encoding='utf-8')
lines = csv.reader(fr,  delimiter='\t')
X_test=[]
y_test=[]

for line in lines:
    if('label' in line[0]):
        continue
    w2i = list(tokenizer(line[1]))
    for subsentence in w2i:
        meanSen.append(len(subsentence))
    meanLine.append(len(w2i))
    X_test.append(w2i)
    y_test.append([int(line[0])])
fr.close()

maxLen = int(np.max(meanSen))
maxSen = int(np.max(meanLine))



print('maxLen:', maxLen)
print('maxSen:', maxSen)
for i  in range(0,len(X_train)):
    for k in range(0, len(X_train[i])):
        n_pad = maxLen - len(X_train[i][k])
        X_train[i][k].extend([0]*n_pad)
    n_pad = maxSen - len(X_train[i])
    for l in range(0, n_pad):
        temp=[]
        temp.extend([0]*maxLen)
        X_train[i].append(temp)

for i  in range(0,len(X_test)):
    for k in range(0, len(X_test[i])):
        n_pad = maxLen - len(X_test[i][k])
        X_test[i][k].extend([0]*n_pad)
    n_pad = maxSen - len(X_test[i])
    for l in range(0, n_pad):
        temp=[]
        temp.extend([0]*maxLen)
        X_test[i].append(temp)


print("pre-trained embedding loading........")

k=0
weights = list()
for i in range(0, len(embed_lookup.wv.vocab)):
    cc = embed_lookup.wv.index2word[i]
    try:
        weights.append(np.ndarray.tolist(embed_lookup[cc]))
    except KeyError:
        weights.append(np.ndarray.tolist(np.random.rand(300,)))
    k+=1                                                                   
weights = np.array(weights, dtype=np.float32)
weights = torch.from_numpy(weights)
weights = torch.FloatTensor(weights)

print("pre-trained embedding loading success....")

print("length of trainset: ", len(X_train))
print("length of testset: ", len(X_test))


class IrisDataset(Dataset):
    
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
                                                    
    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx])
        y = torch.LongTensor(self.labels[idx])
        return x,y

# update args and print
args.embed_num = 300
args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#model = CNN_Text(args,weights)
model = HCL_CLSTM_CLSTM(args,weights,'HCL')
batch = 16
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
data_iter = DataLoader(dataset=IrisDataset(X_train, y_train), batch_size=batch, shuffle=False)
data_test_iter = DataLoader(dataset=IrisDataset(X_test, y_test), batch_size=batch, shuffle=False)
loss_sum =0.


model.cuda()
model.train()
steps=0
criterion = nn.CrossEntropyLoss()  
print("training..............")
for e in range(0, 100):
    iter_bar = tqdm(data_iter, desc='Iter (loss=X.XXX)')
    loss_sum=0
    for tt, samples  in enumerate(iter_bar):
        text, target = samples
        text, target = text.cuda(), target.cuda()
        optimizer.zero_grad()

        logit = model(text)
        loss = criterion(logit.cuda(), target.squeeze().cuda())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, 100, loss_sum/(tt+1)))
    iter_bar_22 = tqdm(data_test_iter, desc='Iter (loss=X.XXX)')
    corrects, avg_loss, alls = 0, 0, 0
    k=1
    for tt, samples in enumerate(iter_bar_22):
        text, target = samples
        text, target = text.cuda(), target.cuda()
        logit = model(text)
        _, y_pred3 = logit.max(1)
        avg_loss += loss.item()
        for i in range(0, len(y_pred3)):
            if(y_pred3[i].item() == target[i]):
                corrects+=1

        iter_bar_22.set_description('Iter (accur=%5.3f)'% float(corrects/(batch*k)))
        k+=1

    avg_loss = avg_loss/k
    accuracy = 100.0 * (float(corrects/(k*batch)))

    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, k))

