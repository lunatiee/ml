import tensorflow as tf
import numpy as np

tf.__version__

w = tf.Variable(2.)
def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:
    z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)


