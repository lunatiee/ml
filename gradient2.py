import tensorflow as tf
import numpy as np


w = tf.Variable(4.0)
b = tf.Variable(1.0)

def hypothesis(x):
    return w*x + b

x_test = [3.5, 5, 5.5, 6]

# print(hypothesis(x_test))
# print(hypothesis(x_test).numpy())

def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y)) #reduce_mean은 차원을 줄이고 평균을 구하는 함수

x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적


optimizer = tf.optimizers.legacy.SGD(0.01)

for i in range(3001):
    with tf.GradientTape() as tape:
        y_pred = hypothesis(x)
        cost = mse_loss(y_pred, y)
    gradients = tape.gradient(cost, [w,b])
    optimizer.apply_gradients(zip(gradients, [w,b]))

    if i % 10 == 0:
        print("epoch : {:3} | w 의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))


print(hypothesis(x_test).numpy())
