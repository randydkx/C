import os
import struct
import numpy as np
import logging.config
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
logging.config.fileConfig(
    '/Users/wenshuiluo/coding/Python/机器学习/CNN实现/config/logging.conf')
# create logger
logger = logging.getLogger('main')

# 持久化开关
TRACE_FLAG = False
# loss曲线开关
LOSS_CURVE_FLAG = True
trace_file = '/Users/wenshuiluo/temp/tmpdata/tmp_data.log'
path_minst_unpack = '/Users/wenshuiluo/coding/Python/深度学习入门与实践/picture/MNIST/raw'

INIT_W = 0.01  # 权值初始化参数
LEARNING_BASE_RATE = 0.1  # 基础学习率
LEARNING_DECAY_RATE = 0.99  # 学习率衰减系数
REG_PARA = 0.5  # 正则化乘数
LAMDA = 1e-4  # 正则化系数lamda
EPOCH_NUM = 50  # EPOCH
MINI_BATCH_SIZE = 100  # batch_size
ITERATION = 1  # 每batch训练轮数
HIDDEN_LAYER_NUM = 500  # size of hidden layer
TYPE_K = 10  # 分类类别


# 设置缺省数值类型
DTYPE_DEFAULT = np.float32

# 有选择地持久化训练结果


def traceMatrix(M, epoch, name):

    if TRACE_FLAG == False:
        return 0
    row = len(M)
    try:
        col = len(M[0])
    except TypeError:
        col = 1
    with open(trace_file, 'a') as file:
        file.write(
            'Epoch[%s]-[%s:%d X %d ]----------------------------------------\n' % (epoch, name, row, col))
        for i in range(row):
            file.write('%s -- %s\n' % (i, M[i]))

# Loss曲线 1图


def showCurves(idx, x, ys, line_labels, colors, ax_labels):
    LINEWIDTH = 2.0
    plt.figure(figsize=(8, 4))
    # loss
    ax1 = plt.subplot(211)
    for i in range(2):
        line = plt.plot(x[:idx], ys[i][:idx])[0]
        plt.setp(line, color=colors[i],
                 linewidth=LINEWIDTH, label=line_labels[i])

    ax1.xaxis.set_major_locator(MultipleLocator(4000))
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.set_xlabel(ax_labels[0])
    ax1.set_ylabel(ax_labels[1])
    plt.grid()
    plt.legend()

    # Acc
    ax2 = plt.subplot(212)
    for i in range(2, 4):
        line = plt.plot(x[:idx], ys[i][:idx])[0]
        plt.setp(line, color=colors[i],
                 linewidth=LINEWIDTH, label=line_labels[i])

    ax2.xaxis.set_major_locator(MultipleLocator(4000))
    ax2.yaxis.set_major_locator(MultipleLocator(0.02))
    ax2.set_xlabel(ax_labels[0])
    ax2.set_ylabel(ax_labels[2])

    plt.grid()
    plt.legend()
    plt.show()


# 加载mnist
def load_mnist_data(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as labelfile:
        # 读取前8个bits
        magic, n = struct.unpack('>II', labelfile.read(8))
        # 余下的数据读到标签数组中
        labels = np.fromfile(labelfile, dtype=np.uint8)

    with open(images_path, 'rb') as imagefile:
        # 读取前16个bit
        magic, num, rows, cols = struct.unpack('>IIII', imagefile.read(16))
        # 余下数据读到image二维数组中，28*28=784像素的图片共60000张（和标签项数一致）
        # reshape 从原数组创建一个改变尺寸的新数组(28*28图片拉直为784*1的数组)
        images = np.fromfile(
            imagefile, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# 输出层结果转换为标准化概率分布，
# 入参为原始线性模型输出y ，N*K矩阵，
# 输出矩阵规格不变


def softmax(y):
    # # 每列都减去该列最大值
    y1 = y - np.max(y, axis=1, keepdims=True)
    # 计算exp
    exp_y = np.exp(y1)
    # 计算softmax得到N*K矩阵
    return exp_y/np.sum(exp_y, axis=1, keepdims=True)

# 计算交叉熵
# 输入为两个 N * K 矩阵,y_为正确答案
# 输出为1* N 的交叉熵数组


def loss_cross_entropy(y_, y):
    # clip限制极值，避免溢出和除0错
    y1 = np.clip(y, 1e-10, 1.0)
    p_logq = y_ * np.log(y1) * -1
    # 分类数固定不变，直接对batch中每个样本的交叉熵取平均，与每行求和后取平均意义一样
    loss_mean = np.mean(p_logq)
    return loss_mean

# 定义执行过程


def main():

    logger.info('start..')
    # 初始化

    # 持久化参数初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    # 类别标签定义，用于构建输出层节点
    LABELS_NUMS = [i for i in range(TYPE_K)]

    # 加载训练数据
    images_ori, labels = load_mnist_data(path_minst_unpack, 'train')
    # 用一部分数据进行训练
    used_size = 1000
    images_ori, labels = images_ori[:used_size], labels[:used_size]
    logger.info('train data loaded')

    # 加载验证数据
    images_v_ori, labels_v = load_mnist_data(path_minst_unpack, 't10k')
    images_v_ori, labels_v = images_v_ori[:used_size], labels_v[:used_size]
    logger.info('10k data loaded')

    # 图像数据归一化
    images = images_ori / 255
    images_v = images_v_ori / 255

    # 模型参数初始化, w：D*H 矩阵， b：H *1 数组
    w = np.sqrt(1/images.shape[1]) * np.random.randn(len(images[0]), HIDDEN_LAYER_NUM)  # D*K
    b = np.zeros(HIDDEN_LAYER_NUM, dtype=DTYPE_DEFAULT)  # 1*K

    # w2 H*K 矩阵 , b2 k*1 数组
    w2 = np.sqrt(1/HIDDEN_LAYER_NUM) * np.random.randn(HIDDEN_LAYER_NUM, len(LABELS_NUMS))
    b2 = np.zeros(len(LABELS_NUMS))

    logger.info('w,b inited..')
    # 训练
    # 样本类别 K
    n_class = TYPE_K
    # 样本范围
    sample_range = [i for i in range(len(labels))]
    valid_range = [i for i in range(len(labels_v))]

    if True == LOSS_CURVE_FLAG:
        cur_p_idx = 0
        curv_x = np.zeros(EPOCH_NUM*100, dtype=int)
        curv_ys = np.zeros((4, EPOCH_NUM*100), dtype=DTYPE_DEFAULT)

    batches_per_epoch = int(np.ceil(len(labels) / MINI_BATCH_SIZE))
    for epoch in range(EPOCH_NUM):
        # 学习率指数衰减
        learning_rate = LEARNING_BASE_RATE * (LEARNING_DECAY_RATE**epoch)
        rest_range = sample_range
        for batch in range(batches_per_epoch):
            # 无放回抽样每次随机抽一个mini-batch进行I轮训练，遍历全部训练sample
            curr_batch_size = min(MINI_BATCH_SIZE, len(rest_range))
            samples = random.sample(rest_range, curr_batch_size)
            rest_range = list(set(rest_range).difference(set(samples)))

            #   输入 N*D
            x = np.array([images[sample]
                          for sample in samples], dtype=DTYPE_DEFAULT)
            #   正确类别 1*K
            values = np.array([labels[sample] for sample in samples])
            # 正确标准编码为onehot encod   N * K
            y_ = np.eye(n_class)[values]

            # 前向传播,得到N*K原始结果
            # 经过隐含层，得到N*H形状的隐含层输出
            hidden_layer = np.maximum(0, x @ w + b)  # ReLU activation
            y = hidden_layer @ w2 + b2  # 和np.dot作用一样 H * K
            # 对原始输出做softmax，规格不变仍为N*K
            softmax_y = softmax(y)
            
            # 每个batch中都对训练误差、验证误差、训练精确度、验证集准确率进行计算
            # train_loss
            corect_logprobs = - \
                np.log(softmax_y[range(curr_batch_size), values])
            data_loss = np.sum(corect_logprobs) / curr_batch_size
            # 正则化项，对每个矩阵求F范数再相加
            reg_loss = REG_PARA * LAMDA * (np.sum(w*w) + np.sum(w2*w2))
            loss = data_loss + reg_loss
            # 测试集acc
            # 使用RELU函数进行激活
            y_v_hidden_layer = np.maximum(0, images_v @ w + b)  # 和np.dot作用一样 N * K
            y_v = y_v_hidden_layer @ w2 + b2
            # 预测结果 1 * 100
            labels_pre = np.argmax(y_v, axis=1)
            accuracy = np.mean(labels_pre == labels_v)
            if True == LOSS_CURVE_FLAG:
                # val loss
                softmax_y_v = softmax(y_v)
                corect_logprobs_v = - \
                    np.log(softmax_y_v[range(len(labels_v)), labels_v])
                data_loss_v = np.sum(corect_logprobs_v) / len(labels_v)
                loss_v = data_loss_v + reg_loss
                # train_acc
                labels_pre_tr = np.argmax(y, axis=1)
                accuracy_tr = np.mean(labels_pre_tr == values)
                # curv_data_x
                curv_x[cur_p_idx] = epoch * ITERATION * \
                    len(labels)/MINI_BATCH_SIZE + \
                    batch * ITERATION + 1
                # train_loss
                curv_ys[0][cur_p_idx] = loss
                # val_loss
                curv_ys[1][cur_p_idx] = loss_v
                # train_acc
                curv_ys[2][cur_p_idx] = accuracy_tr
                # val_acc
                curv_ys[3][cur_p_idx] = accuracy
                cur_p_idx += 1
            #logger.info('epoch %d, loss=%s, loss_v=%s, acc= %s, acc_v = %s' % (epoch + 1, loss, loss_v,accuracy_tr, accuracy))
            logger.info('epoch %d , loss=%s, accuracy = %s' %
                        (epoch, loss, accuracy))
            # 反向传播处理 sigma(z)-y 
            softmax_y[range(curr_batch_size), values] -= 1
            # 计算层级误差
            doutput = softmax_y / curr_batch_size
            delta_w2 = hidden_layer.T @ doutput
            # 正则化部分梯度
            delta_w2 += LAMDA * w2
            delta_b2 = np.sum(doutput, axis=0)
            w2 = w2 - learning_rate * delta_w2
            b2 = b2 - learning_rate * delta_b2
            
            dhidden = doutput @ w2.T
            # backprop the ReLU non-linearity
            # 隐含层节点输出小于零的点被RELU抑制
            dhidden[hidden_layer <= 0] = 0
            delta_w = x.T @ dhidden
            # 正则化部分梯度
            delta_w += LAMDA * w
            delta_b = np.sum(dhidden, axis=0)
            w = w - learning_rate * delta_w
            b = b - learning_rate * delta_b

    # 持久化训练结果
    traceMatrix(w, epoch, 'final_w')
    traceMatrix(b, epoch, 'final_b')
    traceMatrix(w2, epoch, 'final_w2')
    traceMatrix(b2, epoch, 'final_b2')

    # 图示
    if True == LOSS_CURVE_FLAG:
        showCurves(cur_p_idx, curv_x, curv_ys, ['train_loss', 'val_loss', 'train_acc', 'val_acc'], [
                   'y', 'r', 'g', 'b'], ['Iteration', 'Loss', 'Accuracy'])


# 执行
if __name__ == '__main__':
    main()
