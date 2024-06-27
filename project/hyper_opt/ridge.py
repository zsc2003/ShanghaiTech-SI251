# Use for test
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from functools import *

import sys
sys.path.append("../")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge


# Load DataSet
iris = datasets.load_diabetes()
A = iris.data
b = iris.target

scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)

# K-fold
K = 5
KF = KFold(n_splits = K)

A_train_splitted = []
A_test_splitted = []
b_train_splitted = []
b_test_splitted = []


for train_index, test_index in KF.split(A_scaled):
    A_train, A_test = A_scaled[train_index], A_scaled[test_index]
    A_train_splitted.append(A_train)
    A_test_splitted.append(A_test)
    b_train, b_test = b[train_index], b[test_index]
    b_train_splitted.append(b_train)
    b_test_splitted.append(b_test)


out = A_train_splitted[0].shape[1]
I_out = np.eye(out)

def function_objective(y, A_test_splitted, b_test_splitted):
    sum = 0
    for i in range(K):
        y_k = y[i]
        sum += (0.5) * np.sum(np.square(np.dot(A_test_splitted[i], y_k) - b_test_splitted[i])) / A_test_splitted[i].shape[0]
    return sum

weight = np.array([])
# alphas = np.linspace(0,1000,10001)
loss = 10000000
idx = 0
alphas = np.array([1000])
for a in alphas:
    for i in range(K):
        reg = Ridge(alpha=a, fit_intercept=False)
        reg.fit(A_train_splitted[i], b_train_splitted[i])
        w = reg.coef_
        if i == 0:
            weight = np.array([w])
        else:
            weight = np.vstack((weight, [w]))

        y_pred = reg.predict(A_test_splitted[i])

        mse = mean_squared_error(b_test_splitted[i], y_pred)
        print("Mean Squared Error on test set: ", mse / K)

    now_loss = function_objective(weight, A_test_splitted, b_test_splitted) / K
    if loss > now_loss:
        idx = a
        loss = now_loss

print(alphas)
print(a)
print(loss)