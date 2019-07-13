import numpy as np

def Sigmoid(x):
    return 1./(1. + np.exp(-x))

lamda = 1
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])                      # x: 4*2
t = np.array([[0],[1],[1],[0]])                                     # t: 4*1
w1 = np.array([[-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]])                # w1: 2*3
w2 = np.array([[-0.5], [0.5], [-0.5]])                              # w2: 3*1
b1 = np.array([0, 0, 0])

for i in range(100):
    h = Sigmoid(np.dot(x, w1) + b1)                                 # h: 4*3
    y = Sigmoid(np.dot(h, w2))                                      # y: 4*1
    delta_y = np.multiply(y - t, np.multiply(y, (1 - y)))           # delta_y: 4*1
    delta_h = delta_y * np.multiply(w2.T, np.multiply(h, (1 - h)))  # delta_h: 4*3
    w2 = w2 - np.dot(h.T, lamda * delta_y)                          #
    w1 = w1 - np.dot(x.T, lamda * delta_h)
    b1 = b1 - lamda * delta_h
print(y)
