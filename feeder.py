
from math import *
import numpy as np


xs = []
with open("xs", 'r') as fp:
    xs = fp.read().split('\n')
xs = list(map(float, xs))

ys = []
with open("ys", 'r') as fp:
    ys = fp.read().split('\n')
ys = list(map(float, ys))

X_SHAPE = (1)
Y_SHAPE = (1)

def get_shapes():
    return X_SHAPE, Y_SHAPE

def get_feeder():
    n = len(xs)
    for i in range(n // 2):
        yield np.array([xs[i % n]]), np.array([ys[i % n]])

def get_realcalc(x):
    return 3.2414484 * x + 0.12445646
