from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

digits = load_digits()

fig = plt.figure(figsize=(6, 6)) # figure size in inches
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
plt.show()

print(digits.images[0].shape, digits.data[0].shape, digits.target[0])
print(digits.data.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, y_train)
pred = kNN.predict(X_test)

print(metrics.confusion_matrix(y_test, pred))