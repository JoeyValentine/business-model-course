#! /c/Apps/Anaconda3/python
"""
[Title] Understanding Feature Vectors
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-feature_vector.py  
[Description]
  - 교육생 실습용
[History]
  - 2019-05-01 : 최초 작성
[References]
  - 
"""
import numpy as np 

featureVector = np.array([1,2,10])
print("[featureVector] ", featureVector)
print("[Type of featureVector] ", type(featureVector))
print("[Shape of featureVector]", featureVector.shape)

inputVector2D = np.array(featureVector,ndmin=2)
print("[inputVector2D]", inputVector2D)
print("[Shape of inputVector2D]", inputVector2D.shape)

transposedInputVect2D = inputVector2D.T
print("[transposedInputVect2D]", transposedInputVect2D)
print("[Shape of transposedInputVect2D] ", transposedInputVect2D.shape)
