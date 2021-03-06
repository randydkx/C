"""
谱聚类实现
"""

import numpy as np


class KMeans(object):
    def __init__(self, k=3, epochs=100, tol=1e-3, dist_method=None):
        """
        :param k: 聚类簇数量
        :param epochs: 最大迭代次数
        :param tol: 终止条件
        :param dist_method:距离函数，默认欧氏距离
        """
        self.k = k
        self.epochs = epochs
        self.tol = tol
        self.dist_method = dist_method
        if self.dist_method is None:
            self.dist_method = lambda x, y: np.sqrt(np.sum(np.power(x - y, 2)))
        self.cluster_centers_ = {}  # 记录簇中心坐标

    def fit(self, X):
        m = X.shape[0]
        # 初始化
        for idx, data_idx in enumerate(np.random.choice(list(range(m)), self.k, replace=False).tolist()):
            self.cluster_centers_[idx] = X[data_idx]
        # 迭代
        for _ in range(self.epochs):
            C = {}
            for idx in range(self.k):
                C[idx] = []
            for j in range(m):
                best_k = None
                min_dist = np.infty
                for idx in range(self.k):
                    dist = self.dist_method(self.cluster_centers_[idx], X[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_k = idx
                C[best_k].append(j)
            # 更新
            eps = 0
            for idx in range(self.k):
                vec_k = np.mean(X[C[idx]], axis=0)
                eps += self.dist_method(vec_k, self.cluster_centers_[idx])
                self.cluster_centers_[idx] = vec_k
            # 判断终止条件
            if eps < self.tol:
                break

    def predict(self, X):
        m = X.shape[0]
        rst = []
        for i in range(m):
            vec = X[i]
            best_k = None
            min_dist = np.infty
            for idx in range(self.k):
                dist = self.dist_method(self.cluster_centers_[idx], vec)
                if dist < min_dist:
                    min_dist = dist
                    best_k = idx
            rst.append(best_k)
        return np.asarray(rst)

class Spectral(object):
    def __init__(self, n_clusters=None, n_components=None, gamma=None):
        """
        :param n_clusters: 聚类数量
        :param n_components: 降维数量
        :param gamma: rbf函数的超参数
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.gamma = gamma
        if self.n_components is None:
            self.n_components = 10
        if self.gamma is None:
            self.gamma = 1
        if self.n_clusters is None:
            self.n_clusters = 3

    def fit_transform(self, X):
        rows, cols = X.shape
        # 1.构建拉普拉斯矩阵
        W = np.zeros(shape=(rows, rows))
        for i in range(0, rows):
            for j in range(i, rows):
                w = np.exp(-1 * np.sum(np.power(X[i] - X[j], 2)) / (2 * self.gamma * self.gamma))
                W[i, j] = w
                W[j, i] = w
        D = np.diag(np.sum(W, axis=0))
        L = D - W
        # 2.对拉普拉斯矩阵特征分解
        eig_vals, eig_vecs = np.linalg.eig(L)
        sorted_indice = np.argsort(eig_vals)  # 默认升序排序
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        return eig_vecs[:, 0:self.n_components].real

    def fit_predict(self, X):
        # 3.对特征矩阵进行聚类
        transform_matrix = self.fit_transform(X)
        transform_matrix = transform_matrix / np.sqrt(np.sum(np.power(transform_matrix, 2), axis=1, keepdims=True))
        kmeans = KMeans(k=self.n_clusters)
        kmeans.fit(transform_matrix)
        return kmeans.predict(transform_matrix)