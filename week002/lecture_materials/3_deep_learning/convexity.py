import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sig(x):
    return 1/(1+np.exp(-x))

x = np.array([1,2,3,4,5,6])
y = np.array([0,0,0,1,1,1])
xy = np.array([[1,0],[2,0],[3,0],[4,1],[5,1],[6,1]])
n = 100
a = np.linspace(-5, 5, n)
b = np.linspace(-2.5, 2.5, n)
a, b = np.meshgrid(a, b)

cost1 = np.zeros((100, 100)) # cross entropy function
for val in xy:
    _tmp = (val[1] - sig(a + b * val[0]))**2 # error sums of squre function
    cost1 += _tmp

cost2 = np.zeros((100, 100))
for val in xy:
    _tmp = -val[1] * np.log(sig(a + b * val[0])) - (1 - val[1]) * np.log(1 - sig(a + b * val[0]))
    cost2 += _tmp

ax = plt.axes(projection='3d')
ax.contour3D(a, b, cost1, 100, cmap='binary') # satisfies convexity
plt.show()
ax = plt.axes(projection='3d')
ax.contour3D(a, b, cost2, 100, cmap='binary')
plt.show()
