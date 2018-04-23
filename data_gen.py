
import random
from math import *
import numpy as np

from feeder import get_realcalc


xs = []
ys = []

iteration = list(np.arange(0.0, 1.0, 0.00001245))
random.shuffle(iteration)
for x in iteration:
    y = get_realcalc(x * pi)
    xs.append(f"{x}")
    ys.append(f"{y}")

with open("xs", 'w') as fp:
    fp.write('\n'.join(map(str, xs)))

with open("ys", 'w') as fp:
    fp.write('\n'.join(map(str, ys)))
