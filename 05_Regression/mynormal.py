
import mylasso, mylr, myransac, myridge, myrf
import argparse
from time import process_time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

#%%
# df = pd.read_csv('housing.txt', sep='\s+', header=None)
# df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#
#
# X = df.iloc[:, 0:-1]
# print(X.shape)
# #y = df.iloc[:, -1]
# y = df[['MEDV']].values
# print(y.shape)
def normal_equation(X, y):
    # adding a column vector of "ones"
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    y_pred = Xb @ w
    return y_pred

# print('Slope: %.3f' % w[1])
# print('Intercept: %.3f' % w[0])
#
# # print(w)
#
# y_pred = Xb@w
# print(y_pred.shape)
#
# error = mean_squared_error(y, y_pred)
# print('MSE: %.3f' % (error))
#
# r2 = r2_score(y, y_pred)
# print('R^2: %.3f' % (r2))