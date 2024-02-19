# Author: Paul Strebenitzer
# Additional Python file to plot some functions for Bachelors Thesis

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 10)
y = sigmoid(x)
plt.plot(x, y)
plt.title("Sigmoid Function")
plt.savefig('breastcancer_detection_p3.11/Plots/Sigmoid_Function.png')
plt.show()

# Softmax function
def softmax(x):
    return (np.exp(x)/np.exp(x).sum(axis=0, keepdims=True))

x = np.linspace(-5, 5, 10)
y = softmax(x)
plt.plot(x, y)
plt.title("Softmax Function")
plt.savefig('breastcancer_detection_p3.11/Plots/Softmax_Function.png')
plt.show()

# model for roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
names = iris.target_names
X = iris.data
y = iris.target

# only two classes from the iris dataset
X = X[y != 2]
y = y[y != 2]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 500 * n_features)] #augment data

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc, )
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('breastcancer_detection_p3.11/Plots/Example_ROC.png')
plt.show()
