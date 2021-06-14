from typing import Text
import nltk
from torch.nn.modules import conv, padding
from torchtext.legacy import data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from torchtext.vocab import Vectors,GloVe
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam, optimizer
from torchvision import transforms
import torch.nn.functional as F

mytokenize = lambda x: x.split()
TEXT = data.Field(sequential=True,tokenize=mytokenize,
                    include_lengths = True,use_vocab =True,batch_first = True,fix_length=200)
LABEL = data.Field(sequential=False,use_vocab = False,pad_token = None,unk_token=None)
train_test_fields = {
    ("label",LABEL),
    ("text",TEXT)
}
traindata,testdata = data.TabularDataset.splits(
    path="data",format="csv",
    train="imdb_train.csv",fields=train_test_fields,
    test = "imdb_test.csv",skip_header=True
)
print(len(traindata),len(testdata))

ex0 = traindata.examples[0]
print(ex0.label)
print(ex0.text)

train_data,val_data = traindata.split(split_ratio=0.7)
print(len(train_data),len(val_data))

vec = Vectors("glove.6B.100d.txt","data/glove")
TEXT.build_vocab(train_data,max_size=20000,vectors=vec)
LABEL.build_vocab(train_data)
# 训练集中的前十个高频词
print(TEXT.vocab.freqs.most_common(n=10))
print("词典的词数：",len(TEXT.vocab.itos))
print("前十个单词：",TEXT.vocab.itos[:10])
print("类别标签情况：",LABEL.vocab.freqs)

# 定义加载器，将类似长度的实例一起进行批处理
BATCH_SIZE = 12
train_iter = data.BucketIterator(train_data,batch_size=BATCH_SIZE)
val_iter = data.BucketIterator(val_data,batch_size=BATCH_SIZE)
test_iter = data.BucketIterator(testdata,batch_size=BATCH_SIZE)
# 获得一个batch的数据，对数据内容进行介绍
for step,batch in enumerate(train_iter):
    if step>0:
        break
    # 针对一个batch的数据，获得其类别标签
    print("数据的类别标签：\n",batch.label)
    print("数据的尺寸：",batch.text[0].shape)
    print("数据样本数量：",len(batch.text[1]))

class CNN_Text(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_filters,filter_sizes,output_dim,dropout,pad_idx):
        super().__init__()
        # 对文本进行词嵌入
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=n_filters,kernel_size=(fs,embedding_dim)) 
                for fs in filter_sizes
        ])
        # 全连接层和dropout层
        self.fc = nn.Linear(len(filter_sizes * n_filters),output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # 未完待续