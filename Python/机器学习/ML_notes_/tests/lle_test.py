from matplotlib import pyplot as plt
import numpy as np

n = 200
r = np.linspace(0, 1, n)
l = np.linspace(0, 1, n)

t = (3 * np.pi) / 2 * (1 + 2 * r)
x = t * np.cos(t)
y = 10 * l
z = t * np.sin(t)

data = np.c_[x, y, z]

import os
os.chdir('../')

from ml_models.decomposition import LLE

lle = LLE(k=3,n_components=2)
new_data = lle.fit_transform(data)
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.xlim([-0.1,0.2])
plt.show()