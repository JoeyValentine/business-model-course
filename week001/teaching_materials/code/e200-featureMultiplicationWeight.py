#! /c/Apps/Anaconda3/python
"""
[Title] Understanding Multiplication between Features and Weight
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-featureMultiplicationWeight.py  
[Description]
  - 교육생 실습용
[History]
  - 2019-05-01 : 최초 작성
[References]
  - 
"""
import numpy as np

features = np.random.rand(5,3)*10
print("[feature]\n", features)
print("[shape of features] ", features.shape)

Weights = np.random.rand(3,2)
print("[Weights]\n", Weights)
print("[shape of Weights] ", Weights.shape)
dotMultiplication = np.dot(features, Weights)
print(dotMultiplication)

dot_products = np.dot(features, Weights)
print("[dot_products]\n")
print(dot_products)
print("[shape of dot_products]", dot_products.shape)