import enum
from operator import le
from os import EX_UNAVAILABLE, initgroups
from hiddenlayer import canvas
import numpy as np
from numpy.testing._private.utils import break_cycles
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy 
import time
import torch
from torch._C import CompilationUnit
import torch.nn as nn
from torch.nn.modules import padding
from torch.optim import Adam, optimizer
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(
    root = "picture",
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size=64,
    shuffle=False
)

# trainloader中将数据切分成batch大小的数据块，数据块数量就是train_laoder的len
print("train_loader中的batch数量：",len(train_loader))


# 可视化一个batch中的训练数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
    print(b_x)
    print(b_x.size())
    print(b_y.size())
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_label = train_data.classes
    class_label[0] = "T-shirt"
    plt.figure(figsize=(12,5))
    for index in range(len(batch_y)):
        plt.subplot(4,16,index+1)
        plt.imshow(batch_x[index,:,:],cmap=plt.cm.gray)
        plt.title(class_label[batch_y[index]],size=9)
        plt.axis("off")
        # 设置行每张图片之间的间隔
        plt.subplots_adjust(wspace=0.05)
    # plt.show()


test_data = FashionMNIST(
    root="picture",
    train=False,
    download=False
)

test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x,dim=1)
test_data_y = test_data.targets
print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape",test_data_y.shape)

print_flag = True
class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#输入的feature_map数量
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),#经过卷积之后将1*28*28的图片变成16*28*28的图片
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),#经过池化层之后(16*28*28)->(16*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,0),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*6*6,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # if print_flag:
        #     print(x.size(0))
        # 将卷积图层展平
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output

myconvnet = MyConvNet()
print(myconvnet)

# 定义模型的训练过程和验证过程
def train_model(model,traindataloader,train_rate,criterion,optimizer,num_epochs=25):
    # 总的batch数量和用于训练的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-'*10)
        #  每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        # 用于验证的数据，分别是验证集上的误差验证集上的正确数量和验证集的总数
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # 每个epoch对所有的batch进行训练或者测试
        for step,(t_x,t_y) in enumerate(traindataloader):
            if step < train_batch_num:
                # 设置model为训练模式
                model.train()
                output = model(t_x)
                # 按照每行根据输出计算最大的一个值对应的索引，即是函数的预测类别
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,t_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * t_x.size(0)
                train_corrects += torch.sum(pre_lab == t_y.data)
                train_num += t_x.size(0)
            else:
                # model进入评估模式
                model.eval()
                output = model(t_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,t_y)
                val_loss += loss.item() * t_x.size(0)
                val_corrects += torch.sum(pre_lab == t_y.data)
                val_num += t_x.size(0)
        
        # 计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num )
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss:{:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Validation Loss:{:.4f} Validation Acc: {:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))
        # 拷贝最优模型的参数，每个epoch计算一次，并将最优的模型weight保存
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time()-since
        print("train and valiation complete in {:.0f} m {:.0f}s".format(time_use // 60,time_use % 60))
    model.load_state_dict(best_model_wts)
    # 记录训练数据，以便可视化分析
    train_process = pd.DataFrame(
        data = {
            "epoch":range(num_epochs),
            "train_loss_all":train_loss_all,
            "val_loss_all":val_loss_all,
            "train_acc_all":train_acc_all,
            "val_acc_all":val_acc_all
        }
    )
    return model,train_process


# 非空洞情况下卷积神经网络的训练以及可视化
# optimizer = torch.optim.Adam(myconvnet.parameters(),lr=0.0003)
# criterion = nn.CrossEntropyLoss()
# myconvnet,train_process = train_model(myconvnet,train_loader,0.8,criterion,optimizer,num_epochs=25)
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(train_process.epoch,train_process.train_loss_all,"ro-",label="train loss")
# plt.plot(train_process.epoch,train_process.val_loss_all,"bs-",label="val loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("LOSS")
# plt.subplot(1,2,2)
# plt.plot(train_process.epoch,train_process.train_acc_all,"ro-",label="train accuracy")
# plt.plot(train_process.epoch,train_process.val_acc_all,"bs-",label="val accuracy")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# # 对测试集进行预测，可视化查看测试效果
# myconvnet.eval()
# output = myconvnet(test_data_x)
# pre_lab = torch.argmax(output,1)
# acc = accuracy_score(test_data_y,pre_lab)
# print("在测试集上的精度为：",acc)

# conf_mat = confusion_matrix(test_data_y,pre_lab)
# df_cm = pd.DataFrame(conf_mat,index=class_label,columns=class_label)
# heatmap = sns.heatmap(df_cm,annot=True,fmt="d",cmap="YlGnBu")
# heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
# heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
# plt.ylabel('true label')
# plt.xlabel('predicted label')
# plt.show()


# 空洞卷积下CNN的计算
class MyConvDilaNet(nn.Module):
    def __init__(self):
        super(MyConvDilaNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#输入的feature_map数量
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=2
            ),#经过卷积之后将1*28*28的图片变成16*28*28的图片
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),#经过池化层之后(16*28*28)->(16*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,0,dilation=2),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # if print_flag:
        #     print(x.size(0))
        # 将卷积图层展平
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output

# myconvdilanet = MyConvDilaNet()
# print(myconvdilanet)
# optimizer = torch.optim.Adam(myconvdilanet.parameters(),lr=0.0003)
# criterion = nn.CrossEntropyLoss()
# myconvdilanet,train_process = train_model(myconvdilanet,train_loader,0.8,criterion,optimizer,num_epochs=5)
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(train_process.epoch,train_process.train_loss_all,"ro-",label="train loss")
# plt.plot(train_process.epoch,train_process.val_loss_all,"bs-",label="val loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("LOSS")
# plt.subplot(1,2,2)
# plt.plot(train_process.epoch,train_process.train_acc_all,"ro-",label="train accuracy")
# plt.plot(train_process.epoch,train_process.val_acc_all,"bs-",label="val accuracy")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# # 对测试集进行预测，可视化查看测试效果
# myconvnet.eval()
# output = myconvnet(test_data_x)
# pre_lab = torch.argmax(output,1)
# acc = accuracy_score(test_data_y,pre_lab)
# print("在测试集上的精度为：",acc)

# conf_mat = confusion_matrix(test_data_y,pre_lab)
# df_cm = pd.DataFrame(conf_mat,index=class_label,columns=class_label)
# heatmap = sns.heatmap(df_cm,annot=True,fmt="d",cmap="YlGnBu")
# heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
# heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
# plt.ylabel('true label')
# plt.xlabel('predicted label')
# plt.show()


from torchvision.datasets import ImageFolder
from torchvision import models
vgg16 = models.vgg16(pretrained=True)
# 获取特征提取层
vgg = vgg16.features
# 对其特诊提取层进行冻结，不再对其进行更新
for param in vgg.parameters():
    param.requires_grad_(False)

class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel,self).__init__()
        # 提取图像的特征
        self.vgg = vgg
        # 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.vgg(x)
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output

myvggCNN = MyVggModel()
print(myvggCNN)


# 准备新网络需要的数据
# 对训练集进行预处理
train_data_transforms = transforms.Compose(
    [transforms.RandomResizedCrop(244) , ##随机长宽比裁减为244*244
    transforms.RandomHorizontalFlip(),## 按照概率p=0.5进行水平翻转
    transforms.ToTensor(), ## 转化成张量并归一化至[0-1]之间
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)
# 对测试集进行预处理
val_data_transforms = transforms.Compose(
    [transforms.Resize(256),## 对图像重新设置分辨率
    transforms.CenterCrop(224), ## 按照给定的size从中间裁减
    transforms.ToTensor(),## 转化成张量并且归一化
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)

train_data_dir = "data/archive/training"
train_data = ImageFolder(train_data_dir,transform=train_data_transforms)
train_data_loader = Data.DataLoader(train_data,batch_size=32,shuffle=True)
# 读取验证集
val_data_dir = "data/archive/validation"
val_data = ImageFolder(val_data_dir,transform=val_data_transforms)
val_data_loader = Data.DataLoader(val_data,batch_size=32,shuffle=True)

print("验证集样本数：",len(train_data.targets))
print("测试集样本数量：",len(val_data.targets))

# 将一个batch数据可视化

# for step ,(b_x,b_y) in enumerate(train_data_loader):
#     if step > 0:
#         break
#     mean = np.array([0.485,0.456,0.406])
#     std = np.array([0.229,0.224,0.225])
#     plt.figure(figsize=(12,6))
#     for index in range(len(b_y)):
#         plt.subplot(4,8,index+1)
#         image = b_x[index,:,:,:].numpy().transpose((1,2,0))
#         image = image * std + mean
#         # 将小于等于0的设置成0，大于等于1的设置成1
#         image = np.clip(image,0,1)
#         plt.imshow(image)
#         plt.title(b_y[index].data.numpy())
#         plt.axis("off")
#     plt.subplots_adjust(hspace=0.3)
#     plt.show()

import hiddenlayer as hl
# 微调网络
optimizer = torch.optim.Adam(myvggCNN.parameters(),lr=0.003)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
canvas1 = hl.Canvas()
num_epochs = 1
for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-'*10)
        #  每个epoch有两个训练阶段
        train_loss_epoch = 0.0
        train_corrects = 0
        # 用于验证的数据，分别是验证集上的误差验证集上的正确数量和验证集的总数
        val_loss_epoch = 0.0
        val_corrects = 0
        myvggCNN.train()
        # 每个epoch对所有的batch进行训练或者测试
        for step,(t_x,t_y) in enumerate(train_data_loader):
            output = myvggCNN(t_x)
            # 按照每行根据输出计算最大的一个值对应的索引，即是函数的预测类别
            pre_lab = torch.argmax(output,1)
            loss = loss_func(output,t_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * t_x.size(0)
            train_corrects += torch.sum(pre_lab == t_y.data)
        
        # 计算一个epoch的训练损失
        train_loss = train_loss_epoch / len(train_data.targets)
        train_acc = train_corrects.double() / len(train_data.targets)

        # model进入评估模式    
        myvggCNN.eval()
        for step,(val_x,val_y) in enumerate(val_data_loader):
            output = myvggCNN(val_x)
            # 按照每行根据输出计算最大的一个值对应的索引，即是函数的预测类别
            pre_lab = torch.argmax(output,1)
            loss = loss_func(output,val_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss_epoch += loss.item() * val_x.size(0)
            val_corrects += torch.sum(pre_lab == val_y.data)
        
        # 计算一个epoch在验证集上的精度
        val_loss = val_loss_epoch / len(val_data.targets)
        val_acc = val_corrects.double() / len(val_data.targets)
        history1.log(epoch,train_loss = train_loss,val_loss = val_loss,train_acc = train_acc.item(),val_acc = val_acc.item())
        # 可视化网络的训练过程
        with canvas1 : 
            canvas1.draw_plot([history1['train_loss'],history1['val_loss']])
            canvas1.draw_plot([history1['train_acc'],history1['val_acc']])
            