from operator import index
from re import S
from hiddenlayer import canvas
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
import seaborn as sns
import matplotlib.pyplot as plt
import hiddenlayer as hl
from torch.utils.data import dataloader
from torch.utils.data import dataset
from torch.utils.data.sampler import BatchSampler
from torchviz import make_dot
spam = pd.read_csv("data/spambase/spambase.data")
print(spam.head())
# 计算垃圾邮件和废垃圾邮件的数量
print(pd.value_counts(spam.label))
X = spam.iloc[:,0:57].values
y = spam['label'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=123)
scales = MinMaxScaler(feature_range=(0,1))
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.fit_transform(X_test)
colname = spam.columns.values[:-1]
plt.figure()
for item in range(len(colname)):
    plt.subplot(7,9,item+1)
    sns.boxplot(x=y_train,y=X_train_s[:,item])
    plt.title(colname[item])
plt.subplots_adjust(hspace=0.4)
# plt.show()

# 使用全连接很精网络实现垃圾邮件的分类

# MLP分类器部分代码
# class MLPclassfication(nn.Module):
#     def __init__(self):
#         super(MLPclassfication,self).__init__()
#         self.hidden1 = nn.Sequential(
#             nn.Linear(
#                 in_features=57,
#                 out_features=30,
#                 bias=True
#             ),
#             nn.ReLU()
#         )

#         self.hidden2 = nn.Sequential(
#             nn.Linear(30,10),
#             nn.ReLU()
#         )

#         self.classifica = nn.Sequential(
#             nn.Linear(10,2),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         fc1 = self.hidden1(x)
#         fc2 = self.hidden2(fc1)
#         output = self.classifica(fc2)
#         return fc1,fc2,output

# # 创建网络实例以及可视化
# mlpc = MLPclassfication()
# x = torch.randn(1,57).requires_grad_(True)
# y = mlpc(x)
# # dict中包含了网络的权重信息
# print(dict(list(mlpc.named_parameters()) + [('x',x)]))
# Mymlpcvis = make_dot(y,params=dict(list(mlpc.named_parameters()) + [('x',x)]))
# Mymlpcvis.format="png"
# Mymlpcvis.directory=""
# # Mymlpcvis.view()
# # Mymlpcvis.save("全连接网络（分类邮件）.png")

# # 使用标准化之后的数据对样本进行训练
# # 将原数据转化成float32类型
# X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
# Y_train_t = torch.from_numpy(y_train.astype(np.int64))
# X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
# y_test_t = torch.from_numpy(y_test.astype(np.int64))

# train_data = Data.TensorDataset(X_train_t,Y_train_t)
# # 声明数据加载器
# train_loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=64,
#     shuffle=True
# )

# optimizer = Adam(params=mlpc.parameters(),lr=0.01)
# loss_func = nn.CrossEntropyLoss()
# # 记录训练过程中的指标
# history1 = hl.History()
# # 声明画布
# canvas1 = hl.Canvas()
# print_step = 25
# for epoch in range(15):
#     for step,(t_x,t_y) in enumerate(train_loader):
#         _,_,output = mlpc(t_x)
#         train_loss = loss_func(output,t_y)
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#         # 计算迭代次数
#         niter = epoch * len(train_loader)+step+1
#         if niter % print_step == 0 :
#             _,_,output = mlpc(X_test_t)
#             # 将最大的一个置为1，
#             _,prediction = torch.max(output,1)
#             test_accuracy = accuracy_score(y_test_t,prediction)
#             history1.log(niter,train_loss=train_loss,test_accuracy=test_accuracy)

#             # 使用两个图像可视化训练的精度
#             with canvas1:
#                 canvas1.draw_plot(history1['train_loss'])
#                 canvas1.draw_plot(history1['test_accuracy'])

# # 将最终的模型应用到测试集中，测试分类的准确率
# _,_,output = mlpc(X_test_t)
# _,prediction = torch.max(output,1)
# test_accuracy = accuracy_score(y_test_t,prediction)
# print("在测试集上的准确度：",test_accuracy)

# # 可视化网络的中间层输出
# _,test_fc2,_ = mlpc(X_test_t)
# # 输出是[1151,10]:表示一共有1151个测试样本，每个样本有十个特征输出
# print("test_fc2.shape:",test_fc2.shape)
# # 对输出进行降维和可视化
# test_fc2_tsne = TSNE(n_components=2).fit_transform(test_fc2.data.numpy())

# # 将特征进行可视化
# plt.figure(figsize=(8,6))
# # 设置画布的最小最大值区间分别是第一维和第二维的最小最大值
# plt.xlim(min(test_fc2_tsne[:,0]),max(test_fc2_tsne[:,0]))
# plt.ylim(min(test_fc2_tsne[:,1]),max(test_fc2_tsne[:,1]))
# # 将真实标签为0和1的分别在画布中画出
# plt.plot(test_fc2_tsne[y_test==0,0],test_fc2_tsne[y_test==0,1],"bo",label="0")
# plt.plot(test_fc2_tsne[y_test==1,0],test_fc2_tsne[y_test==1,1],"rd",label="1")
# plt.legend()
# plt.title("test_fc2_tsne")
# plt.show()    




# MLP回归器的构建
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch.nn.functional as F
housedata = fetch_california_housing()
X_train,X_test,y_train,y_test = train_test_split(housedata.data,housedata.target,test_size = 0.25,random_state = 123)
# 数据标准化处理
scale = StandardScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.fit_transform(X_test)
# 转化成数据表，便于探索数据的分布等
housedatadf = pd.DataFrame(data=X_train_s,columns=housedata.feature_names)
housedatadf['target']=y_train
print(housedatadf.head())

# 可视化相关系数的热力图
datacor = housedatadf.corr()
plt.figure(figsize=(8,6))
ax = sns.heatmap(datacor,square=True,annot=True,fmt='.3f',linewidths=.5,cmap='YlGnBu',
cbar_kws={'fraction':0.046,"pad":0.03})
# plt.show()
print("训练数据的维度：",X_train_s.shape)
train_xt = torch.from_numpy(X_train_s.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test_s.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))
train_data = Data.TensorDataset(train_xt,train_yt)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True
)

# 构建回归模型
class MLPregreession(nn.Module):
    def __init__(self):
        super(MLPregreession,self).__init__()
        self.hidden1 = nn.Linear(in_features=8,out_features=100,bias=True)
        self.hidden2 = nn.Linear(100,100)
        self.hidden3 = nn.Linear(100,50)
        self.predict = nn.Linear(50,1)
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:,0]

mlpreg = MLPregreession()
print(mlpreg)

optimizer = torch.optim.SGD(params=mlpreg.parameters(),lr=0.01)
loss_func = nn.MSELoss()
train_loss_all = []
for epoch in range(30):
    train_loss = 0
    train_num = 0
    for step,(tx,ty) in enumerate(train_loader):
        # print(step,tx.shape,ty.shape)
        # tx是torch.Size类型，使用.size(i)函数获取第i维的维度
        if epoch==0:
            print(tx.size(0,))
        output = mlpreg(tx)
        loss = loss_func(output,ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算该batch训练的所有样本的损失，loss.item中返回的是单个样本的损失，*tx.size()返回的是
        train_loss+=loss.item() * tx.size(0)
        # 该批次训练的样本总数量
        train_num += tx.size(0)
    train_loss_all.append(train_loss / train_num)

plt.figure(figsize=(8,6))
plt.plot(train_loss_all,"ro-",label="train loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("Loss")
# plt.show()

# 对测试集进行预测并绘制预测曲线
prediction = mlpreg(test_xt)
prediction = prediction.data.numpy()
mae =mean_absolute_error(y_test,prediction)
print("在测试集上的平均绝对误差",mae)
# 按照y值对正确的数据进行排序，将预测数据用散点图绘制在正确数据的同一张图中
index = np.argsort(y_test)
plt.figure(figsize=(8,6))
plt.plot(np.arange(len(y_test)),y_test[index],"r",label="Original y")
plt.scatter(np.arange(len(prediction)),prediction[index],s=3,c='b',label = "prediction")
plt.legend()
plt.grid()
plt.xlabel("index")
plt.ylabel("Y")
plt.show()