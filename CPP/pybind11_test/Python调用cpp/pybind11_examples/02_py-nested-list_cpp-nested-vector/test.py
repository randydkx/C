
import example
import numpy as np

A = np.arange(10).reshape(2,5)

B = example.modify(A)

print(B)

print(example.modify(A.tolist()))
