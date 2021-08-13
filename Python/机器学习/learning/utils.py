import numpy as np
import copy
from scipy import special
import math
import matplotlib.pyplot as plt


def sign(x):
    """
    符号函数
    :param x:
    :return:
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def sigmoid(x2):
    """
    sigmoid函数
    :param x2:
    :return:
    """
    x = copy.deepcopy(x2)
    if type(x) is int:
        x = 20.0 if x > 20.0 else x
        x = -100.0 if x < -100.0 else x
    else:
        # 避免下溢
        x[x > 20.0] = 20.0
        # 避免上溢
        x[x < -100.0] = -100.0
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    softmax函数
    :param x:
    :return:
    """
    if x.ndim == 1:
        return np.exp(x) / np.exp(x).sum()
    else:
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def entropy(x, sample_weight=None):
    """
    计算熵
    :param x:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])

    # 计算熵
    ent = .0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        ent += -p_i * math.log(p_i)
    return ent


def cond_entropy(x, y, sample_weight=None):
    """
    计算条件熵:H(y|x)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    ent = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_x = x[x_index]
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        p_i = 1.0 * len(new_x) / x_num
        ent += p_i * entropy(new_y, new_sample_weight)
    return ent


def muti_info(x, y, sample_weight=None):
    """
    互信息/信息增益:H(y)-H(y|x)
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return entropy(y, sample_weight) - cond_entropy(x, y, sample_weight)


def info_gain_rate(x, y, sample_weight=None):
    """
    信息增益比
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return 1.0 * muti_info(x, y, sample_weight) / (1e-12 + entropy(x, sample_weight))


def gini(x, sample_weight=None):
    """
    计算基尼系数 Gini(D)
    :param x:
    :param sample_weight:
    :return:
    """
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])

    # 计算gini系数
    gini_value = 1.0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        gini_value -= p_i * p_i
    return gini_value


# def cond_gini(x, y, sample_weight=None):
#     """
#     计算条件gini系数:Gini(y,x)
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     # x中元素个数
#     x_num = len(x)
#     # 如果sample_weight为None设均设置一样
#     if sample_weight is None:
#         sample_weight = np.asarray([1.0] * x_num)
#     # 计算
#     gini_value = .0
#     for x_value in set(x):
#         x_index = np.where(x == x_value)
#         new_x = x[x_index]
#         new_y = y[x_index]
#         new_sample_weight = sample_weight[x_index]
#         p_i = 1.0 * len(new_x) / x_num
#         gini_value += p_i * gini(new_y, new_sample_weight)
#     return gini_value


# def gini_gain(x, y, sample_weight=None):
#     """
#     gini值的增益
#     """
#     x_num = len(x)
#     if sample_weight is None:
#         sample_weight = np.asarray([1.0] * x_num)
#     return gini(y, sample_weight) - cond_gini(x, y, sample_weight)


# 计算Gini(D,A)
def gini_D_conditioned_A(x,y,index_value,sample_weight):
    """计算Gini(D,A)

    Args:
        x (ndarray): 数据矩阵
        y (ndarray): 标签
        index_value (int): 属性索引
        sample_weight (list): 样本权重

    Returns:
        gini指数
    """     
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    indices1 = np.where(x == index_value)
    indices2 = np.where(x != index_value)
    D1 = x[indices1]
    D2 = x[indices2]
    y1 = y[indices1[0]]
    y2 = y[indices2[0]]
    gini_value = len(D1) / x_num * gini(y1,sample_weight[indices1[0]]) + len(D2) / x_num * gini(y2,sample_weight[indices2[0]])
    return gini_value

def square_error(x, sample_weight=None):
    """
    平方误差
    :param x:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    # x_mean = np.mean(x)
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_mean = np.dot(x, sample_weight / np.sum(sample_weight))
    error = 0.0
    for index in range(0, x_num):
        error += (x[index] - x_mean) * (x[index] - x_mean) * sample_weight[index]
    return error


def cond_square_error(x, y, sample_weight=None):
    """
    计算按x分组的y的误差值
    :param x:
    :param y:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    error = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        error += square_error(new_y, new_sample_weight)
    return error


def square_error_gain(x, y, sample_weight=None):
    """
    平方误差带来的增益值
    :param x:
    :param y:
    :param sample_weight:
    :return:
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return square_error(y, sample_weight) - cond_square_error(x, y, sample_weight)


def gaussian_1d(x, u, sigma):
    """
    一维高斯概率分布函数
    :param x:
    :param u:
    :param sigma:
    :return:
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma + 1e-12) * np.exp(-1 * np.power(x - u, 2) / (2 * sigma ** 2 + 1e-12))


def gaussian_nd(x, u, sigma):
    """
    高维高斯函数
    :param x:对x矩阵的每行进行高斯计算
    :param u:
    :param sigma:
    :return:
    """
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    return 1.0 / (np.power(2 * np.pi, x.shape[1] / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(
        np.sum(-0.5 * (x - u).dot(np.linalg.inv(sigma)) * (x - u), axis=1))


def dirichlet(u, alpha):
    """
    狄利克雷分布
    :param u: 随机变量
    :param alpha: 超参数
    :return:
    """
    # 计算归一化因子
    beta = special.gamma(np.sum(alpha))
    for alp in alpha:
        beta /= special.gamma(np.sum(alp))
    rst = beta
    # 计算结果
    for idx in range(0, len(alpha)):
        rst *= np.power(u[idx], alpha[idx] - 1)
    return rst


def wishart(Lambda, W, v):
    """
    wishart分布
    :param Lambda:随机变量
    :param W:超参数
    :param v:超参数
    :return:
    """
    # 维度
    D = W.shape[0]
    # 先计算归一化因子
    B = np.power(np.linalg.det(W), -1 * v / 2)
    B_ = np.power(2.0, v * D / 2) * np.power(np.pi, D * (D - 1) / 4)
    for i in range(1, D + 1):
        B_ *= special.gamma((v + 1 - i) / 2)
    B = B / B_
    # 计算剩余部分
    rst = B * np.power(np.linalg.det(Lambda), (v - D - 1) / 2)
    rst *= np.exp(-0.5 * np.trace(np.linalg.inv(W) @ Lambda))
    return rst


def St(X, mu, Lambda, v):
    """
    学生t分布
    :param X: 随机变量
    :param mu: 超参数
    :param Lambda: 超参数
    :param v: 超参数
    :return:
    """
    n_sample, D = X.shape
    return special.gamma(D / 2 + v / 2) * np.power(np.linalg.det(Lambda), 0.5) * np.power(
        1 + np.sum((X - mu) @ Lambda * (X - mu), axis=1) / v, -1.0 * D / 2 - v / 2) / special.gamma(v / 2) / np.power(
        np.pi * v, D / 2)


"""
绘制决策边界
"""


def plot_decision_function(X, y, clf, support_vectors=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    # 绘制支持向量
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolor='red')


"""
绘制等高线
"""


def plot_contourf(data, func, lines=3):
    n = 256
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), n)
    X, Y = np.meshgrid(x, y)
    C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), lines, colors='g', linewidth=0.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(data[:, 0], data[:, 1])

    
if __name__=='__main__':
    x = gaussian_nd(np.array([[1,2],[3,4]]),np.array([1,1]),np.array([[1,0],[0,1]]))
    print(x)# [0.09653235 0.00023928]
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print(a*b)
    


# 数据进行分箱操作，
#  x < a -- 0
#  a < x < b --- 1
# 依次类推，将连续的属性取值离散化
class DataBinWrapper(object):
    def __init__(self,max_bin = 10):
        super().__init__()
        self.max_bin = max_bin
        self.XrangeMap = None
    
    def fit(self,X):
        _ , n_features = X.shape
        self.XrangeMap = [[] for _ in range(n_features)]
        for index in range(0,n_features):
            # 找出对应的属性
            tmp = sorted(X[:,index])
            for percent in range(1,self.max_bin):
                # 找到相应的分位数10%-90%分位数
                percent_value = np.percentile(tmp,(1.0 * percent / self.max_bin) * 100 // 1)
                self.XrangeMap[index].append(percent_value)
            self.XrangeMap[index] = sorted(list(self.XrangeMap[index]))
    
    def transform(self,X):
        # 如果只有一个样本，将x的每一维都进行分箱
        # np.digitize返回的是给的元素在列表中的索引区间，从1开始
        if X.ndim == 1:
            return np.asarray([np.digitize(X[i],self.XrangeMap[i]) for i in range(X.shape[0])])
        else:
            return np.asarray([np.digitize(X[:,i],self.XrangeMap[i]) for i in range(X.shape[1])]).T