from ast import Add
import torch
import numpy as np
def AdditionFc(a, b):
    print(np.random.randn(1,2))
    print("Now is in python module")
    print("{} + {} = {}".format(a, b, a+b))
    return a + b

if __name__ == "__main__":
    AdditionFc(1, 2)