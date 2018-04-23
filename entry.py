
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
cost = tf.abs(Y - model)
optimizer = tf.train.AdamOptimizer(0.00025).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    i = 0
    for x, y in feeder:
        _, cost_value = session.run(
            (optimizer, cost),
            feed_dict={X: x, Y: y})

        i += 1
        if i % 1024 == 0:
            print(f"step: {i}, cost: {cost_value[0]:.12f}, w: {session.run(w1)}, b: {session.run(b1)}")

    test_xs = np.arange(0, 1.0, 0.12)
    random.shuffle(test_xs)
    for x in test_xs:
        y = session.run(model, feed_dict={X: [x]})[0]
        a = get_realcalc(x)
        cost_value = session.run(cost, feed_dict={X: [x], Y: [y]})[0]
        print(f"x: {x:.4f}, y: {y:.4f}, a: {a:.4f}, cost: {cost_value:.4f}")
