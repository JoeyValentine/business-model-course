#! /c/Apps/Anaconda3/python
"""
[Title] Understanding Activation
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-activation.py  
[Description]
  - 교육생 실습용
[History]
  - 2019-05-01 : 최초 작성
[References]
  -
"""
import numpy as np 
from matplotlib import pyplot as plt 


def sigmoid(feature):
	return 1 / (1 + np.exp(-feature))

feature = np.linspace(-10, 10, 1000)
# print("[feature] ", feature)

plt.plot(feature, sigmoid(feature))
plt.text(4,0.8, r'$\sigma(feature)=\frac{1}{1+e^{-feature}}$', fontsize=20)
plt.show()

