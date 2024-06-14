import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from functools import *

# plz change the path to the location of your project
import sys
sys.path.append("../")
sys.path.append("D:/ddn")
from ddn.basic.node import *

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
 

class RidgeNode(AbstractDeclarativeNode):
    
    def __init__(self, n, K, A_train_splitted, b_train_splitted, I_out):
        super().__init__(1, n)
        self.K = K
        self.A_train_splitted = A_train_splitted
        self.b_train_splitted = b_train_splitted
        self.I_out = I_out
        
    # x is miu
    # y is [beta_1^T,beta_2^T,...,beta_K^T]
    def objective(self, x, y):
        sum = 0
        for i in range(self.K):
            sum += 0.5 * np.sum(np.square(np.dot(self.A_train_splitted[i],y[i])-self.b_train_splitted[i])) + 0.5 * x * np.linalg.norm(y[i],ord = 2)
        return sum

    def solve(self, x):
        # TODO
        result = np.array([])
        for i in range(self.K):
            if i == 0:
                result = sci.linalg.solve(np.dot(self.A_train_splitted[i].T,self.A_train_splitted[i]) + x * self.I_out, np.dot(self.A_train_splitted[i].T, self.b_train_splitted[i]))
            else:
                result = np.vstack((result, sci.linalg.solve(np.dot(self.A_train_splitted[i].T,self.A_train_splitted[i]) + x * self.I_out, np.dot(self.A_train_splitted[i].T,self.b_train_splitted[i]))))
            # print(result.shape)
        return result, None

    def gradient(self, x, y=None, ctx=None):
        # TODO
        if y is None:
            y, ctx = self.solve(x)
        
        result = np.array([])
        for i in range(self.K):
            if i == 0:
                result = (-1) * sci.linalg.solve(np.dot(self.A_train_splitted[i].T,self.A_train_splitted[i]) + x * self.I_out, y[i])
            else:
                result = np.vstack((result, (-1) * sci.linalg.solve(np.dot(self.A_train_splitted[i].T,self.A_train_splitted[i]) + x * self.I_out, y[i])))
            # print(result.shape)
        
        return result







class ParameterOptimizer:
    
    def __init__(self, step_size, tol, max_iters, dataset, K):
        self.step_size = step_size
        self.tol = tol
        self.max_iters = max_iters
        self.A = dataset.data  
        self.b = dataset.target  
        # 数据预处理：标准化特征
        scaler = StandardScaler()
        A_scaled = scaler.fit_transform(self.A)

        # K-fold
        self.K = 5
        KF = KFold(n_splits = self.K)

        self.A_train_splitted = []
        self.A_test_splitted = []
        self.b_train_splitted = []
        self.b_test_splitted = []

        for train_index, test_index in KF.split(A_scaled):
            A_train, A_test = A_scaled[train_index], A_scaled[test_index]
            self.A_train_splitted.append(A_train)
            self.A_test_splitted.append(A_test)
            b_train, b_test = self.b[train_index], self.b[test_index]
            self.b_train_splitted.append(b_train)
            self.b_test_splitted.append(b_test)
            # print("X_train: ", A_train.shape)
            # print("X_test: ", A_test.shape)
            

        self.out = self.A_train_splitted[0].shape[1]
        self.I_out = np.eye(self.out)

    def function_objective(self, y):
        sum = 0
        for i in range(self.K):
            y_k = y[i]
            sum += (0.5) * np.sum(np.square(np.dot(self.A_test_splitted[i], y_k) - self.b_test_splitted[i]))
        return sum

    def derivative_objective_y(self, y):
        result = np.array([])
        for i in range(self.K):
            if i == 0:
                result = np.dot((np.dot(y[i].T, self.A_test_splitted[i].T) - self.b_test_splitted[i].T), self.A_test_splitted[i])
            else:
                result = np.vstack((result, np.dot((np.dot(y[i].T, self.A_test_splitted[i].T) - self.b_test_splitted[i].T), self.A_test_splitted[i])))
            # print(result.shape)
        return result

    def simpleGradientDescent(self, node, miu_init, verbose=False):
        """
        An example of gradient descent for a simple bi-level optimization problem of the form:
            minimize_{miu} 
            subject to y = argmin_z f(miu, z)

        Returns the solution x found and learning curve (objective function J per iteration).

        `node.solve`, which solves the (lower-level) optimization problem to produce $y$ given $x$, and
        `node.gradient`, which computes $\text{D}y$ given $x$ and optional $y$.

        """
        # assert b.shape[0] == node.dim_y
        # miu = miu_init.copy() if miu_init is not None else np.zeros((node.dim_x,))
        cnt = 0
        # miu = np.array([[miu_init]])
        miu = miu_init
        all_miu = []
        gradient = []
        axis_x = []
        history = []
        loss = []
        print("Start Iteration")
        for i in range(self.max_iters):
            # solve the lower-level problem and compute the upper-level objective
            y, _ = node.solve(miu)
            history.append(self.function_objective(y))
            if verbose: print("{:5d}: {}".format(i, history[-1]))
            if (len(history) > 2) and (history[-2] - history[-1]) < self.tol:
                print("End Iteration")
                print("Weight = {}".format(y))
                
                break
            
            # print(derivative_objective_y(y, A_test_splitted, b_test_splitted).shape)
            # compute the gradient of the upper-level objective with respect to x via the chain rule
            dJdx = 0 
            for j in range(self.K):
                dJdx += np.dot((self.derivative_objective_y(y)[j]).T, node.gradient(miu, y)[j])

            # take a step in the negative gradient direction
            miu -= self.step_size * dJdx
            all_miu.append(miu)
            gradient.append(abs(dJdx))
            cnt += 1
            axis_x.append(cnt)
            loss.append(self.function_objective(y) / (self.K * self.A_test_splitted[0].shape[0]))

        plt.figure(1)
        plt.plot(axis_x, all_miu)
        plt.xlabel("iteration time")
        plt.ylabel("hyperparameter")

        plt.figure(2)
        plt.plot(all_miu, gradient)
        plt.xlabel("hyperparameter")
        plt.ylabel("abs(gradient)")
        plt.axhline(y=0, color='r', linestyle = '--')
        # print(gradient[-1])

        plt.figure(3)
        plt.plot(all_miu, loss)
        plt.xlabel("hyperparameter")
        plt.ylabel("loss function")
        # print(loss[-1])

        print("Iteration time: ", cnt)
        print("Loss = {}".format(loss[-1]))
        print("Gradient = {}".format(gradient[-1]))
        return miu, history
    
def solve_problem():
    optimizer = ParameterOptimizer(1.0e-1, 1.0e-8, 500000, datasets.load_iris(), 5)
    node = RidgeNode(optimizer.K * optimizer.out, optimizer.K, optimizer.A_train_splitted, optimizer.b_train_splitted, optimizer.I_out)
    miu_init = 0.1
    x, history_gd = optimizer.simpleGradientDescent(node, miu_init)
    print("Gradient descent to give x = {}".format(x))
    plt.show()
    return
    

if __name__ == "__main__":
   solve_problem()