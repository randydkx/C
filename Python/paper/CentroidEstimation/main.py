import numpy as np
import scipy.io as scio
import  matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale

class Algorithm_Binary(object):
    def __init__(self, eta, lr = 0.001,epochs = 100, beta = 0.01, loss = 'logistic',lamb = 0.001):
        self.loss_type = loss
        if loss == 'logistic':
            # centroid factor c in algorithm
            self.a = -1/2
        else:
            AssertionError( "not implemented error")
        self.eta = eta
        self.lamb = lamb
        self.beta = beta
        self.d = None
        self.w = None
        self.miu = None
        self.lr = lr
        self.epochs = epochs
    
    def _gradient_binary(self, X, y):
        grad = np.zeros_like(self.w)
        N = X.shape[0]
        for xi, _ in zip(X, y):
            h = np.dot(self.w.T, xi)[0]
            xi = xi.reshape(-1,1)
            # protect from overflowing
            if h >= 0:
                grad += 1/2 *(1 + (2 * np.exp(-h) - 2 * np.exp(-2 * h)) / (1 + 2 * np.exp(-h) + np.exp(-2 * h))) * xi
            else:
                grad += 1/2 *(-1 + (2 * np.exp(h) + 2 * np.exp(2 * h)) / (1 + 2 * np.exp(h) + np.exp(2 * h))) * xi
        return grad * (1 / N) + self.lamb * self.w + self.a / (1 - 2 * self.eta) * self.miu
    
    def _miu_step(self):
        self.miu = self.miu_hat + self.inv_sigma_hat @ self.w * \
            np.sqrt(self.beta / (self.w.T @ self.inv_sigma_hat @ self.w)[0][0])
    
    def _w_step(self, X, y):
        self.w = self.w - self.lr * self._gradient_binary(X, y)
    
    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_
    
    def _grad_pure_logistic(self, X, y):
        grad = np.zeros_like(self.w)
        for xi,yi in zip(X, y):
            h = yi * np.dot(self.w.T, xi)[0]
            if h >= 0:
                grad += np.exp(-h) / (1 + np.exp(-h)) * (-yi) * xi.reshape(-1,1)
            else:
                grad += 1 / (1 + np.exp(h)) * (-yi) * xi.reshape(-1,1)
            
            return  grad / X.shape[0] + self.lamb * self.w
        
    def pure_logistic(self, X, y):
        n = X.shape[0]
        self.w = np.random.randn(X.shape[1] + 1).reshape(-1,1)
        X = self._preprocess_data_X(X)
        for i in range(self.epochs):
            self.w = self.w - self.lr * self._grad_pure_logistic(X, y)
    
    def _loss(self, y, y_proba):
        ret = 0
        for yi, ti in zip(y, y_proba):
            ret += 1/2 * np.log(2 + np.exp(-yi * ti) + np.exp(yi * ti))
        
        return ret / (y.shape[0]) + 1/2 * self.lamb * (self.w.T @ self.w)[0][0] + self.a / (1 - 2 * self.eta) * (self.w.T @ self.miu)[0][0]  
    
    def _get_miu(self, X, y, K = 3):
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        size = int(X.shape[0] / K)
        _map={}
        for i in range(K):
            if i == K-1:
                _map[i] = index[i * size : ]
            else:
                _map[i] = index[i * size : (i + 1) * size]
        miu = np.zeros((K, X.shape[1],1))
        r = np.zeros((K, K))
        for i in range(K):
            miu[i,:] = np.mean(y[_map[i]].reshape(-1,1) * X[_map[i],:], axis = 0).reshape(X.shape[1], 1).reshape(-1,1)
        i_star = None
        medi_score = np.inf
        for i in range(K):
            for j in range(K):
                r[i][j] = np.power(miu[i] - miu[j],2).sum()
            medi = np.median(r[i])
            if medi < medi_score:
                medi_score = medi
                i_star = i
        return miu[i_star]
    
    def train(self, X, y, eps = 1e-5):
        n = X.shape[0]
        # y = np.array(y)
        # self.ss = StandardScaler()
        # self.ss.fit(X)
        # _X = self.ss.transform(X)
        _X = self._preprocess_data_X(X)
        self.miu_hat = self._get_miu(_X, y, K = 3)
        self.miu_tmp = np.mean(y.reshape(-1,1) * _X, axis = 0).reshape(-1, 1)
        self.sigma_hat = (1 / n / n) * _X.T @ _X - ( 1 / n ) * self.miu_tmp @ self.miu_tmp.T
        self.inv_sigma_hat = np.linalg.inv(self.sigma_hat + 0.001 * np.eye(self.sigma_hat.shape[0]))
        self.d = self.sigma_hat.shape[0]
        # column vector
        self.w = np.random.randn(self.d).reshape(-1,1)
        self.miu = np.zeros_like(self.w)
        self.loss_list = []
        for epoch in range(self.epochs):
            self._miu_step()
            self._w_step(_X, y)
            self.loss_list.append(self._loss(y, self._predict_proba(_X, self.w)))
            if epoch > 2 and np.abs(self.loss_list[-1] - self.loss_list[-2]) < eps:
                print('stop iteration with thresholding')
                break

    def predict(self, X):
        # X = self.ss.transform(X)
        X = self._preprocess_data_X(X)
        y_pred = self._predict_proba(X, self.w)
        return np.where(y_pred >= 0.5, 1, -1)
    
    def _predict_proba(self, X, w):
        z = self._z(X, w)
        return self._sigmoid(z)
    
    def _z(self, X, w):
        return np.dot(X, w)

    def _sigmoid(self, z):
        idx = z >= 0
        ret = np.zeros_like(z)
        # avoid overflowing
        ret[idx] = 1. / (1. + np.exp(-z[idx]))
        ret[~idx] = np.exp(z[~idx]) / (1. + np.exp(z[~idx])) 
        return ret
    
def random_data(x1,y1,x2,y2,x3,y3):
    x1, y1 = x1, y1
    x3, y3 = x3, y3
    x2, y2 = x2, y2
    sample_size = 1000
    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3
    return x,y

def noisify(X, y, eta, random_state = 0):
    y_list = y.copy()
    P = np.array([[1 - eta, eta], [eta, 1-eta]])
    noisy_labels = y_list.copy()
    y_list[ y_list == -1 ] = 0 
    flipper = np.random.RandomState()
    for (idx, label) in enumerate(y_list):
        flipped = flipper.multinomial(1, P[label, :], 1)[0]
        noisy_labels[idx] = np.where(flipped == 1)[0]
    error_ratio = np.mean((noisy_labels != y_list))
    print(f'true error ratio : {error_ratio}')
    noisy_labels[ noisy_labels == 0 ] = -1
    return noisy_labels

def synthetic():
    x_positive, y_positive = random_data(-100, -100, -100, 100, 100, 100)
    x_negative, y_negative = random_data(-40, -100, 100, 20, 100, -100)
    x_data = np.concatenate([x_positive, x_negative])
    y_data = np.concatenate([y_positive, y_negative])
    true_Y = np.concatenate([np.ones_like(x_positive, dtype=np.int), np.ones_like(x_negative, dtype=np.int) * (-1) ])

    X = np.c_[x_data, y_data]
    
    #=== plot==================================================================================================================================
    # plt.figure(figsize=(20, 20))
    # plt.subplot(4,4,1)
    # plt.scatter(X[true_Y == 1, 0], X[true_Y == 1, 1], c='', edgecolors='blue', facecolor='none', marker='o', s=12,
    #             linewidths=1)
    # plt.scatter(X[true_Y == -1, 0], X[true_Y == -1, 1], c='red', marker='+')
    # plt.axis([-100, 100, -100, 100])
    # plt.title('initial data')
    # # plt.show()
    # plt.subplot(1,4,2)
    # plt.scatter(X[noisy_Y == 1, 0], X[noisy_Y == 1, 1], color='', edgecolors='blue', facecolor='none', marker='o', s=12,
    #             linewidths=1)
    # plt.scatter(X[noisy_Y == -1, 0], X[noisy_Y == -1, 1], color='red', marker='+')
    # plt.axis([-100, 100, -100, 100])
    # plt.title('noisy data')
    
    
    test_iter = 4
    acc_list = np.zeros((test_iter,))
    for i in range(test_iter):
        # size = int(np.sqrt(test_iter))
        # noisy rate
        eta = 0.2
        noisy_Y = noisify(X, true_Y, eta)
        X_train, X_test, y_train, y_test = train_test_split(X, noisy_Y, test_size=0.3)

        alg = Algorithm_Binary(eta=eta)

        alg.train(X_train, y_train)
        # alg.pure_logistic(X_train,y_train)
        y_pred = alg.predict(X).reshape(-1)
        accuracy_rate = accuracy_score(true_Y, y_pred)

        print(f'X.shape = {X.shape} -- y.shape = {true_Y.shape} -- eta = {eta}  '+f'accuracy:{accuracy_rate}')
        acc_list[i] = accuracy_rate

        plt.subplot(test_iter, 3,i * 3 + 1)
        plt.scatter(X[true_Y == 1, 0], X[true_Y == 1, 1], edgecolors='blue', facecolor='none', marker='o', s=12,linewidths=1)
        plt.scatter(X[true_Y == -1, 0], X[true_Y == -1, 1], c='red', marker='+')
        plt.title('original')
        plt.axis([-100, 100, -100, 100])
        
        plt.subplot(test_iter, 3,i * 3 + 2)
        plt.scatter(X[noisy_Y == 1, 0], X[noisy_Y == 1, 1], edgecolors='blue', facecolor='none', marker='o', s=12,linewidths=1)
        plt.scatter(X[noisy_Y == -1, 0], X[noisy_Y == -1, 1], c='red', marker='+')
        plt.title('noisy')
        plt.axis([-100, 100, -100, 100])
        
        plt.subplot(test_iter, 3,i * 3 + 3)
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], edgecolors='blue', facecolor='none', marker='o', s=12,linewidths=1)
        plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', marker='+')
        plt.title('prediction')
        plt.axis([-100, 100, -100, 100])
    # plt.savefig(f'pic_{eta}_{np.mean(acc_list)}.png', bbox_inches='tight')
    plt.show()
    print(f'total runs = {acc_list.__len__()} -- mean : {np.mean(acc_list)}')

def train_mode(data, eta = 0.1, lr = 0.01, K = 5):
    Xdata = data['x'][0][0]
    label = data['t'][0][0]
    sampleCapacity, _ = Xdata.shape
    label = np.array(label).reshape(-1)
    dirty_label = noisify(Xdata, label, eta)
    
    index = np.arange(sampleCapacity)
    np.random.shuffle(index)
    
    lamb_list = [np.power(np.float(2),i) / sampleCapacity  for i in range(-12, 3) ]
    beta_list = [np.power(np.float(2),i) / sampleCapacity  for i in range(-12, 13) ]
    eta_list = [0.1, 0.2, 0.3, 0.4]
    best_param = None
    best_score = -np.inf
    for lamb in lamb_list:
        for beta in beta_list:
            accuracy_rate = 0        
            # k-fold cross-validation
            for i in range(1, K + 1):
                selected = np.array([False] * sampleCapacity)
                selected[int((i - 1) * sampleCapacity / K):int(i * sampleCapacity / K)] = True
                X_test = Xdata[index[selected]]
                X_train = Xdata[index[~selected]]
                model = Algorithm_Binary(eta, lr=lr, lamb = lamb, beta = beta)
                X_train_std = minmax_scale(X_train)
                X_test_std = minmax_scale(X_test)
                y_train = dirty_label[index[~selected]]
                model.train(X_train_std, y_train)
                true_test_label = label[index[selected]]
                pred = model.predict(X_test_std)
                accuracy_rate += accuracy_score(true_test_label,pred)
            print((lamb, beta, accuracy_rate / K))
            if accuracy_rate / K > best_score:
                best_score = accuracy_rate / K
                best_param = (lamb, beta, best_score)
    print(f'eta = {eta}  -- best_param = {best_param}')

def UCI():
    benchmarks = scio.loadmat("benchmarks.mat")
    breastCancer = benchmarks['breast_cancer']
    diabetis = benchmarks['diabetis']
    # thyroid = benchmarks['thyroid']
    german = benchmarks['german']
    heart = benchmarks['heart']
    # image = benchmarks['image']
    # australian = benchmarks['australian']
    splice = benchmarks['splice']
    train_mode(heart)
    pass

if __name__=='__main__':
    synthetic()
    # UCI()