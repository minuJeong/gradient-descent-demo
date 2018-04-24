
import math
import random

import tensorflow as tf
import numpy as np

from feeder import get_shapes
from feeder import get_feeder
from feeder import get_realcalc


feeder = get_feeder()

x_shape, y_shape = get_shapes()
X = tf.placeholder(shape=x_shape, dtype=tf.float32)
Y = tf.placeholder(shape=y_shape, dtype=tf.float32)
w1 = tf.Variable(random.random(), dtype=tf.float32)
b1 = tf.Variable(random.random(), dtype=tf.float32)
model = w1 * X + b1
cost = tf.square(model - Y)
gd_optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
rmsprop_optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
adam_optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    i = 0

    cost_stat = []
    for xs, ys in feeder:
        _, cost_value = session.run(
            (gd_optimizer, cost),
            feed_dict={X: xs, Y: ys})
        cost_stat.append(cost_value[0])

        if i % 10 ** 3 == 0:
            mininum_cost = min(cost_stat)
            maximum_cost = max(cost_stat)
            average_cost = sum(cost_stat) / len(cost_stat)
            print(
                f"[{i // 10 ** 3}k] cost: {cost_value[0]:.7f}, w: {session.run(w1)}, b: {session.run(b1)}\n" +
                f"\tcost: {mininum_cost:.7f} ~ {maximum_cost:.7f}, avr: {average_cost:.7f}")
            cost_stat = []

            if average_cost < (0.1):
                print(f"reached average target cost: {cost_value[0]:.7f}, at step: {i}")
                print(f"\tW: {session.run(w1):.4f}, B: {session.run(b1):.4f}")
                break
        i += 1

    test_xs = np.random.random(x_shape)
    test_ys = session.run(model, feed_dict={X: test_xs})
    test_as = [get_realcalc(x) for x in test_xs]
    test_cost = session.run(tf.square(X - Y), feed_dict={X: test_ys, Y: test_as})
    print(
        f" x: {list(test_xs)}\n " +
        f"y: {list(test_ys)}\n " +
        f"a: {list(test_as)}\n " +
        f"cost: {sum(test_cost) / len(test_cost)}")
