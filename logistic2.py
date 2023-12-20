import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x-0.5)
y2 = sigmoid(x)
y3 = sigmoid(x+0.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'c', linestyle='--')
plt.plot([0,0], [1.0,0.0],':')
plt.title('Sigmoid Function')
plt.show()