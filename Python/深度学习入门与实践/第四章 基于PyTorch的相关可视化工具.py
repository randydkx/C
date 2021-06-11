import enum
from posixpath import relpath
from graphviz.backend import view_unixoid
from hiddenlayer import canvas
from sklearn.utils import shuffle
import torch
from torch import optim 
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
import torchvision
from torchvision import datasets
import torchvision.utils as vutils
from torch.optim import SGD, optimizer
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root='picture',
    # 只使用训练数据
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
# 定义数据加载器
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size=128,
    shuffle=True
)

# 准备需要使用的测试数据集
test_data = torchvision.datasets.MNIST(
    root='picture',
    train=False,
    download=False
)

# 为数据添加一个通道维度，将像素值转化到[0-1]之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x,dim=1)
# 获取训练数据的标签
test_data_y = test_data.targets
print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # 定义卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,## 输入的feature map
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32*7*7,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        # 分类层
        self.out = nn.Linear(64,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积图层展平
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        output = self.out(x)
        return output

MyconvNet = ConvNet()
print(MyconvNet)

# 将网络结构可视化以及保存到本地
import hiddenlayer as hl
hl_graph = hl.build_graph(MyconvNet,torch.zeros([1,1,28,28]))
hl_graph.theme = hl.graph.THEMES["blue"].copy()
# hl_graph.save("hl.png",format="png")

# 使用make_dot可视化网络
from torchviz import make_dot
x = torch.randn(1,1,28,28).requires_grad_(True)
y = MyconvNet(x)
Myconvnetvis = make_dot(y,params=dict(list(MyconvNet.named_parameters()) + [('x',x)]))
# 将myconvnet保存成图片
Myconvnetvis.format = "png"
Myconvnetvis.directory = ""
# Myconvnetvis.view()


# 使用tensorboardX进行可视化
# from tensorboardX import SummaryWriter
# SumWriter = SummaryWriter(logdir="log")
# optimizer = torch.optim.Adam(MyconvNet.parameters(),lr=0.0003)
# loss_func =nn.CrossEntropyLoss()
# train_loss = 0
# print_step = 100
# for epoch in range(5):
#     for step,(b_x,b_y) in enumerate(train_loader):
#         output = MyconvNet(b_x)
#         loss = loss_func(output,b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss = train_loss+loss
#         # 计算迭代次数
#         niter = epoch * len(train_loader) + step + 1
#         # 经过print_step步骤之后做记录
#         if niter % print_step == 0:
#             SumWriter.add_scalar("train_loss",train_loss.item() / niter,global_step=niter)
#             # 计算在测试集上的精度
#             output = MyconvNet(test_data_x)
#             _,pre_lab = torch.max(output,1)
#             acc = accuracy_score(test_data_y,pre_lab)
#             # 为日志添加测试集上的精度
#             SumWriter.add_scalar("test acc",acc.item(),niter)
#             # 为日志添加训练数据的可视化图像，使用当前batch的图像
#             # 将一个batch的数据进行预处理
#             b_x_im = vutils.make_grid(b_x,nrow=12)
#             SumWriter.add_image('train image sample',b_x_im,niter)
#             # 使用直方图可视化网络中的参数分布
#             for name,param in MyconvNet.named_parameters():
#                 SumWriter.add_histogram(name,param.data.numpy(),niter)


# 使用hiddenLayer库实现可视化
import hiddenlayer as hl
import time
MyconvNet = ConvNet()
optimizer = torch.optim.Adam(MyconvNet.parameters(),lr=0.0003)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
# 使用canvas进行可视化
canvas1 = hl.Canvas()
print_step = 100
for epoch in range(5):
    for step , (b_x,b_y) in enumerate(train_loader):
        output = MyconvNet(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        canprint = True
        if step % print_step == 0:
            output = MyconvNet(test_data_x)
            _,predict = torch.max(output,1)
            # if canprint:
            #     print(predict)
            #     canprint = False
            acc = accuracy_score(test_data_y,predict)
            history1.log((epoch,step),
            train_loss = loss,
            test_acc = acc,
            hidden_weight = MyconvNet.fc[2].weight
            )
            # 可视化网络的训练过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_acc"])
                canvas1.draw_image(history1["hidden_weight"])

