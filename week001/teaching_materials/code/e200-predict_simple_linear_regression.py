#! /c/Apps/Anaconda3/python
"""
[Title] Understanding Learning Process
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-predict_simple_linear_regression.py  
[Description]
  - 교육생 실습용
[History]
  - 2019-05-01 : 최초 작성
  - 2019-05-05 : Naming 표준화 개선
[References]
  - https://www.tensorflow.org
  - https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.htm
"""
from matplotlib import pyplot as plt
feature = 10.0
Weight = 1.5
bias = 0.3
predictedLabel = Weight*feature + bias 
print('[Predicted Label] Weight({:4.1f})*feature({:4.1f}) + bias({:4.1f}) = '.format(Weight,feature,bias), predictedLabel)

import numpy as np
feature = [10.0, 20.0, 30.0]
label = [10,12,18]

Weight = 1.5
bias = 0.5
print('[feature] ',feature)
print('[data type of feature]',type(feature)) 
print("========== Weight Adjustment [Step 0] ==========")
predictedLabel = np.dot(Weight,feature) + bias
print("[Predicted Labels] ", predictedLabel)
print('[data type of prdictedLabel]',type(predictedLabel)) 
for i in range(len(feature)):
    print('[Initial Predicted Label] Weight({:4.1f})*feature({:4.1f}) + bias({:4.1f}) = '.format(Weight, feature[i],bias), predictedLabel[i])
plt.subplot(1,5,1)
plt.title("Step0")
plt.plot(feature, predictedLabel)
plt.scatter(feature, label)
def f_total_sum(feature, Weight, bias):
    total_sum = 0.0
    for i in range(len(feature)):
        total_sum += Weight * feature[i] + bias
    return total_sum
total_sum = f_total_sum(feature=feature, Weight=Weight, bias=bias)
print('[total estimated sum] ',total_sum)

print('[actual label]', ','.join(str(item) for item in label), '[total actual sum]',sum(label))

def f_Loss_MSE(feature, Weight, bias, label):
    total_error = 0.0
    for i in range(len(feature)):
        total_error += (label[i] - (Weight * feature[i] + bias))**2
    lossMSE = total_error / len(feature)
    return lossMSE

lossMSE = f_Loss_MSE(feature=feature, Weight=Weight, bias=bias, label = label)

print("[Loss] ", lossMSE)

plt.subplot(1,5,2)
plt.title("Step0")
plt.plot(feature, predictedLabel)
plt.scatter(feature, label)
class SLR(object):
    def f_weight_adjust(self, feature, Weight, bias, label, learningRate):
        self.feature = feature
        self.Weight = Weight
        self.bias = bias
        self.label = label
        self.learningRate = learningRate
        self.deltaWeight = 0
        self.deltaBias = 0
        self.numObs = len(self.label)
        for i in range(self.numObs):
            self.deltaWeight -= 2*self.feature[i]*(self.label[i] - (self.Weight * self.feature[i] + self.bias))
            self.deltaBias -= 2*(self.label[i] - (self.Weight * self.feature[i] + self.bias))
        self.Weight  -= (self.deltaWeight / self.numObs) * self.learningRate
        self.bias == (self.deltaBias / self.numObs) * self.learningRate
        return self.Weight, self.bias
slr1 = SLR()
learningRate = 0.0005
print("========== Weight Adjustment [Step 2] ==========")
Weight, bias = slr1.f_weight_adjust(feature=feature, Weight=Weight, bias=bias, label=label, learningRate=learningRate)
print('[Step1]','Weight=', Weight, 'bias=', bias)
predictedLabel = np.dot(Weight, feature) + bias
print("[PredictedValue] ", predictedLabel)
lossMSE_learning_step_1 = f_Loss_MSE(feature=feature, Weight=Weight, bias=bias, label = label)
print("Loss-MSE",lossMSE, "->", lossMSE_learning_step_1)

plt.subplot(1,5,3)
plt.title("Step2")
plt.plot(feature, predictedLabel)
plt.scatter(feature, label)

print("========== Weight Adjustment [Step 3] ==========")
Weight, bias = slr1.f_weight_adjust(feature=feature, Weight=Weight, bias=bias, label=label, learningRate=learningRate)
print('[Step2]','Weight=', Weight, 'bias=', bias)
predictedLabel = np.dot(Weight, feature) + bias
print("[PredictedValue] ", predictedLabel)
lossMSE_learning_step_2 = f_Loss_MSE(feature=feature, Weight=Weight, bias=bias, label = label)
print(lossMSE, "->", lossMSE_learning_step_1, "->", lossMSE_learning_step_2)
plt.subplot(1,3,3)
plt.title("Step3")
plt.plot(feature, predictedLabel)
plt.scatter(feature, label)
plt.show()

print("========== Weight Adjustment [Step 4] ==========")
Weight, bias = slr1.f_weight_adjust(feature=feature, Weight=Weight, bias=bias, label=label, learningRate=learningRate)
print('[Step3]','Weight=', Weight, 'bias=', bias)
predictedLabel = np.dot(Weight, feature) + bias
print("[PredictedValue] ", predictedLabel)
lossMSE_learning_step_3 = f_Loss_MSE(feature=feature, Weight=Weight, bias=bias, label = label)
print(lossMSE, "->", lossMSE_learning_step_1, "->", lossMSE_learning_step_2, "->", lossMSE_learning_step_3)
plt.subplot(1,5,4)
plt.title("Step3")
plt.plot(feature, predictedLabel)
plt.scatter(feature, label)
plt.show()