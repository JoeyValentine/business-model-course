import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv(w):
    partial_b = (-1) * np.sum(y - sigmoid(w[0] + w[1] * x))
    partial_w = (-1) * np.sum((y - sigmoid(w[0] + w[1] * x)) * x)
    return np.array([partial_b, partial_w])

x = np.array([1,2,3,4,5,6])
y = np.array([0,0,0,1,1,1])
w = np.array([0.5,0.5])

np.set_printoptions(suppress=True, precision=3)
alpha = 0.1
for i in range(5000):
    w = w - alpha * deriv(w)
    if i % 1000 == 0:
        print(w[0], w[1])
        yhat = sigmoid(w[0] + w[1] * x)
        print(yhat)