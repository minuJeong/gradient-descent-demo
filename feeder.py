
from random import random
from math import *


xs = []
with open("xs", 'r') as fp:
    xs = fp.read().split('\n')
xs = list(map(float, xs))

ys = []
with open("ys", 'r') as fp:
    ys = fp.read().split('\n')
ys = list(map(float, ys))

BATCH_SIZE = 10
X_SHAPE = (BATCH_SIZE,)
Y_SHAPE = (BATCH_SIZE,)

def get_shapes():
    return X_SHAPE, Y_SHAPE

def get_feeder():
    n = len(xs)
    i = 0
    xds = None
    yds = None
    while True:
        xds, yds = [], []
        for batch in range(BATCH_SIZE):
            idx = (i + batch) % n
            xds.append(xs[idx])
            yds.append(ys[idx])
        yield xds, yds
        i += BATCH_SIZE

def get_realcalc(x):
    w = random() * 0.4 + 8.32
    b = random() * 0.3 + 2.1
    return w * x + b
