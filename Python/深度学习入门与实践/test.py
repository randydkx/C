import numpy as np
x = np.array([3,2,1])
index = np.argsort(x)
print(x)
print(index)
print(x[index])
print(241 * 64)
import torch 
x = torch.tensor([[1],[2]])
print(x)
print(x.size())
print(x.squeeze().numpy())
import nltk

import ssl

try:

    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:

    pass

else:

    ssl._create_default_https_context = _create_unverified_https_context

nltk.download_shell()

