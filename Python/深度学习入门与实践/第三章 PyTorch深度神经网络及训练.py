import torch
print(torch.tensor([1,2]).dtype)
A = torch.arange(6.0).reshape(2, 3)
B = torch.linspace(0, 10, 6).reshape(2, 3)
# dim=0:按照列方向进行堆砌
# dim=1:按照行方向进行堆砌
C = torch.cat((A, B), dim=1)
print(C)
A = torch.tensor(float("nan"))
# 比较两个值是否相等，是否将nan认为相同
print(torch.allclose(A,A,equal_nan=False))
print(torch.allclose(A,A,equal_nan=True))

A = torch.tensor([1,2,3,4,5,6])
B = torch.arange(1,7)
# unsqueeze：增加一个维度
C = torch.unsqueeze(B,dim=0)
print(C)
# eq:是否有相同的元素，equal：是否有相同的size和元素
print(torch.eq(A,B))
print(torch.eq(A,C))
print(torch.equal(A,C))
print(torch.equal(A,B))

# ge:逐元素比较大小
print(torch.ge(A,B))
# 逐元素判断是否是缺失值
print(torch.isnan(torch.tensor([0,1,float("nan")])))

print("\ntensor之间的加减乘除运算")
A = torch.arange(6.0).reshape(2,3)
B = torch.linspace(10,20,steps=6).reshape(2,3)
print(A*B)
print(A/B)
print("逐元素整除取整：",B//A)

# 将超过指定值或者小于指定值的元素设置为指定值
print("张量的裁剪：根据最小值或者最大值：")
print(torch.clamp_max(A,3))
print(torch.clamp_min(A,3))

# 矩阵计算
print("求矩阵的积：")
print("A:",A)
print("A.T:",torch.t(A))
print(A.matmul(A.T))

print("求逆矩阵和矩阵的迹：")
print(torch.inverse(torch.rand(3,3)))
print(torch.trace(torch.rand(3,3)))

print("统计相关的计算：")
A = torch.tensor([312,41,1,3,54,23])
print(torch.max(A))
print(torch.argmax(A))

print("张量的排序：")
print(torch.sort(A))
print(torch.sort(A,descending=True))
print("排序之后的位置索引：")
print(torch.argsort(A))

print("取出tensor中前k大的元素以及位置：")
print(torch.topk(A,4))
print("A:",A)
print("取值大小是第k小的元素以及位置：",torch.kthvalue(A,k=1))

print("\n自动微分系统:")
x = torch.tensor([[1,2],[3,4]],dtype=float,requires_grad=True)
y = torch.sum(x**2+2*x+1)
print("x:",x)
print("y:",y)
# 计算y在x的每个位置上的梯度
y.backward()
print(x.grad)

print("torch.nn模块测试：")
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
myim = Image.open("picture/pic.jpg")
myimgray = np.array(myim.convert("L"),dtype=np.float32)
# plt.figure(figsize=(6,6))
# plt.imshow(myimgray,cmap=plt.cm.gray)
# plt.show()

# 测试nn中的卷积层模块
imh,imw = myimgray.shape

imgray_convert = torch.from_numpy(myimgray.reshape(1,1,imh,imw))
print(imgray_convert.shape)
kersize = 5
ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
ker[2,2]=24
print(ker)
ker = ker.reshape(1,1,kersize,kersize)
print(ker)
conv2d = nn.Conv2d(1,2,(kersize,kersize),bias=False)
conv2d.weight.data[0]=ker
imconv2dout = conv2d(imgray_convert)
imconv2dout_im = imconv2dout.data.squeeze()
print("卷积之后的尺寸：",imconv2dout_im.shape)
# 可视化卷积之后图像
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(imconv2dout_im[0],cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(imconv2dout_im[1],cmap=plt.cm.gray)
plt.axis("off")
# plt.show()

# 测试nn中的池化层模块
maxpool2 = nn.MaxPool2d(2,stride=2)
pool2out = maxpool2(imconv2dout)
pool2out_im = pool2out.squeeze()
print(pool2out_im.shape)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(pool2out_im[0].data,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(pool2out_im[1].data,cmap=plt.cm.gray)
plt.axis("off")
# plt.show()

# 进行平均值池化
maxpool2 = nn.AvgPool2d(2,stride=2)
pool2out = maxpool2(imconv2dout)
pool2out_im = pool2out.squeeze()
print(pool2out_im.shape)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(pool2out_im[0].data,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(pool2out_im[1].data,cmap=plt.cm.gray)
plt.axis("off")
# plt.show()

# 数据准备和相关操作
from sklearn.datasets import load_boston ,load_iris
import torch.utils.data as Data
# 读取回归的数据
boston_X,boston_y = load_boston(return_X_y=True)
print(boston_X.dtype)
print(boston_y.dtype)

train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
print(train_xt.dtype)
print(train_yt.dtype)
train_data = Data.TensorDataset(train_xt,train_yt)
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size=64,
    shuffle=True
)
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
    print("b_x.shape:",b_x.shape)
    print("b_y.shape:",b_y.shape)
    print("b_x.dtype:",b_x.dtype)
    print("b_y.dtype:",b_y.dtype)

# 加载和预处理图像数据
# from torchvision.datasets import FashionMNIST
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# train_data = FashionMNIST(
#     root="/Users/wenshuiluo/coding/Python/深度学习入门与实践/picture",
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True
# )
# train_loader = Data.DataLoader(
#     dataset = train_data,
#     batch_size=64,
#     shuffle=True,
#     num_workers=2,
# )
# print("train_loader的batch数量：",len(train_loader))

# 加载文本数据
import torchtext


# 构建网络
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        # 定义隐藏层
        self.hidden = nn.Sequential(nn.Linear(13,10),nn.ReLU(),)
        # 定义预测回归层
        self.regression=nn.Linear(10,1)
    # 定义网络的前向传播路径
    def forward(self,x):
        x = self.hidden(x)
        output = self.regression(x)
        return output

testnet = TestNet()
print(testnet)

optimizer = torch.optim.Adam(testnet.parameters(),lr=0.001)

conv1 = nn.Conv2d(3,16,3)
torch.manual_seed(12)
# 重新设置conv1的权重
nn.init.normal_(conv1.weight,mean=0,std=1)
plt.figure(figsize=(12,8))
plt.hist(conv1.weight.data.numpy().reshape((-1,1)),bins=30)
# plt.show()

# 加载数据
from torch.optim import SGD
from sklearn.preprocessing import StandardScaler
import pandas as pd
boston_X,boston_y = load_boston(return_X_y=True)
ss = StandardScaler(with_mean=True,with_std=True)
boston_Xs = ss.fit_transform(boston_X)
# 将X，y训练数据转化成张量
train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
# 整合X&y 
train_data = Data.TensorDataset(train_xt,train_yt)
# 定义数据加载器，将训练数据集批量处理
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)

# 构建网络的方法一
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel,self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(
            in_features=13,
            out_features=10,
            bias=True
        )
        self.active1 = nn.ReLU()
        # 第二个隐藏层
        self.hidden2 = nn.Linear(10,10)
        self.active2 = nn.ReLU()
        # 定义预测回归层
        self.regression = nn.Linear(10,1)
    # 定义前向传播路径
    def forward(self,x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.regression(x)
        return output

mlp1 = MLPmodel()
print(mlp1)
optimizer = SGD(mlp1.parameters(),lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []
for epoch in range(30):
    for step,(b_x,b_y) in enumerate(train_loader):
        # 获得输出以及计算均方根误差
        output = mlp1(b_x).flatten()
        train_loss = loss_func(output,b_y)
        # 将梯度清零
        optimizer.zero_grad()
        # 误差反向传播计算梯度与更新梯度
        train_loss.backward()
        optimizer.step()
        train_loss_all.append(train_loss.item())
plt.figure()
plt.plot(train_loss_all,'r-')
plt.title("Train loss per iteration")
# plt.show()


# 定义和训练网络的方式二
class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2,self).__init__()
        # 通过续贯模型建立网络
        self.hidden=nn.Sequential(
            nn.Linear(13,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
        )
        # 建立预测回归层
        self.regression = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
mlp2 = MLPmodel2()
print(mlp2)
# 定义优化器和损失函数
optimizer = SGD(mlp2.parameters(),lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []
for epoch in range(30):
    for step,(b_x,b_y) in enumerate(train_loader):
        output = mlp2(b_x).flatten()
        train_loss = loss_func(output,b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_all.append(train_loss.item())
plt.figure()
plt.legend('方案二')
plt.plot(train_loss_all,'b-')
# plt.show()
torch.save(mlp2,"model/mlp2.pkl")
print('\nloadmodel：')
mlpload = torch.load("model/mlp2.pkl")
print(mlpload)