import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = [[-0.5,-0.5,+0.3,-0.1],[-0.5,+0.5,-0.5,+1.0]]
# x = [[-0.5,-0.5],[-0.5,+0.5],[+0.3,-0.5],[-0.1,+1.0]]
y = [1,1,0,0]
plt.scatter(x[0][:2],x[1][:2],c='b')
plt.scatter(x[0][2:],x[1][2:],c='r')
w0 = -0.5
w1 = -2
b = 0
x1 = np.linspace(-1,1.2)
y = -w1/w0*x1-b/w1
plt.plot(x1,y)
plt.xlim([-0.8,0.8])
plt.ylim([-1.2,1.2])
plt.show()
