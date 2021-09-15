import numpy as np
from datetime import datetime
import sys

from numpy.core.fromnumeric import shape

class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters, np.random.uniform(low,high,size=(m,n)) -> matrix: m * n
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def softmax(self,x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def forward_propagation(self, x):
        # hidden states is h, prediction is y_hat
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        y_hat = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            x_t = np.array(x[t]).reshape(-1,1)
            h[t] = (self.U.dot(x_t) + self.W.dot(h[t-1].reshape(-1,1))).reshape(-1)
            o_t = self.V.dot(h[t])
            y_hat[t] = self.softmax(o_t)
        return y_hat, h
  
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        y, h = self.forward_propagation(x)
        return np.argmax(y, axis=1)

    def calculate_total_loss(self, x, labels):
        total_L = 0
        # For each sentence...
        for i in np.arange(len(labels)):
            y_hat, h = self.forward_propagation(x[i])
            total_L += -1 * sum([np.log(y_pred.T.dot(y_true)) for y_pred,y_true in zip(y_hat,np.array(labels[i]))])
        return total_L
    
    def calculate_loss(self, x, labels):
        # Divide the total loss by the number of training examples 
        N = np.sum([len(label_i) for label_i in labels])
        return self.calculate_total_loss(x,labels)/N

    def bptt(self, x, label):
        T = len(label)
        # Perform forward propagation
        y_hat, h = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        # delta_y -> dLdy: y_hat_t - y_t
        delta_y = np.zeros(y_hat.shape)
        # For each output backwards...
        for t in np.arange(T - 1,-1,-1):
            delta_y[t] = y_hat[t] - np.array(label[t])
            dLdV += delta_y[t].reshape(-1,1) @ h[t].T.reshape(1,-1)
            # Initial delta_t calculation when t is T
            if t == T - 1:
                delta_t = np.diag(1 - np.power(h[t],2)) @ self.V.T @ delta_y[t].reshape(-1,1)
            else:
                delta_t = np.diag(1 - np.power(h[t],2)) @ (self.V.T @ delta_y[t].reshape(-1,1) + self.W.T @ delta_t.reshape(-1,1))
            dLdW += delta_t @ h[t - 1].reshape(1,-1)
            dLdU += delta_t @ np.array(x[t]).reshape(1,-1)
        return dLdU, dLdV, dLdW

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, label, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, label)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{time} Loss after num_examples_seen {num_examples_seen} epoch {epoch}, current loss is {loss}')
            # 在验证集上的精度变低就调整学习率，这是一种超参数调节方法
            if(len(losses)>1 and losses[-1][1]>losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print("Setting learning rate to %f" % learning_rate)
                
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def get_index_and_vocabTable(sentences):
    word2idx = {}
    idx2word = {}
    size = 0
    for sentence in sentences:
        for word in sentence:
            if word2idx.__contains__(word) == False:
                word2idx[word] = size
                idx2word[size] = word
                size += 1    
    return word2idx,idx2word,size

def transform(sentences,word2idx):
    ret = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            tmp.append(word2idx[word])
        ret.append(tmp)
        
    return ret
def transform_back(sentences,idx2word):
    sentences = np.array(sentences)
    ret = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            index = np.argwhere(word == 1)[0][0]
            tmp.append(idx2word[index])
        ret.append(' '.join(tmp))
    return ret
    
if __name__=='__main__':
    s1 = '你 好 李 焕 英'.strip().split()
    s2 = '再 见 夏 洛 特'.strip().split()
    word2idx,idx2word,vocab_size = get_index_and_vocabTable([s1,s2])
    indices = transform([s1,s2],word2idx=word2idx)
    x_sample = np.identity(vocab_size)[indices,:]
    print(transform_back(x_sample,idx2word=idx2word))
    labels = np.identity(vocab_size)[(np.array(indices) + 1) % vocab_size,:]

    rnn = RNN(10)
    train_with_sgd(rnn,x_sample,labels)