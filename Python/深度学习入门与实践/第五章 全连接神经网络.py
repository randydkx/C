import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
import seaborn as sns
import hiddenlayer as hl
from torchiviz import make_dot
spam = pd.read_csv("data/spambase/spambase.data",header=)
print(spam)