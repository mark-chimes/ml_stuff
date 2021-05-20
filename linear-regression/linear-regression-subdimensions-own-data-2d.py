# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.linalg import lstsq
import random
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm

seed = 444
random.seed(seed)

xlim = [-1, 1]
ylim = [-1, 1]

#%% Generate Data
def generatingFunction(x, y): 
    return x*y - x**2 +x - 2*y**2 + y

N = 1000
mu, sigma = 0, 0.1 # mean and standard deviation
sx = np.random.normal(mu, sigma, N)
sy = np.random.normal(mu, sigma*2, N)
sz_diff = np.random.normal(mu, sigma, N)
target_sz = generatingFunction(sx, sy)
sz = target_sz + sz_diff

#%% Linear Regression
num_train = 900

X = np.column_stack((sx, sy))
X_train = X[:num_train,:]
X_test = X[num_train:,:]

y_train = sz[:num_train]
y_test = sz[num_train:]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
y_test_pred = regr.predict(X_test)
y_all_pred = regr.predict(X)

#%% Regression Evaluation

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_test_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_test_pred))


#%%  Least Squares

col1 = np.ones(N)
desmat = np.column_stack((col1, sx, sy))
sol, res, rnk, sing = lstsq(desmat, sz)


#%% Define Plotting Functions 
   
def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X')
    h = plt.ylabel('Y')
    h.set_rotation(0)

#%% Flat Plots
plt.suptitle('Input data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sx, sy, marker='.', color='black')
plt.show()

#%% Training Data Plots

plt.suptitle('Inputs with Chosen Training Points')
standardFlatLimitsAndLabels(plt)
plt.scatter(sx, sy,  marker='o',  color='black')
plt.scatter(X_train[:,0], X_train[:,1], marker='1', color='lightblue')
plt.show()

# Plot the projection 

# Training Data Projection 1
plt.suptitle('Projection of Outputs vs X of Full Dataset and Training Points')
plt.scatter(sx, sz,  marker='o', color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('X')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_train[:,0], y_train,  marker='1', color='blue')
plt.show()

# Training Data Projection 2
plt.suptitle('Projection of Outputs vs Y of Full Dataset and Training Points')
plt.scatter(sy, sz,  marker='o',  color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('Y')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_train[:,1], y_train,  marker='1',  color='blue')
plt.show()

#%% Test Data Plots

plt.suptitle('Inputs with Chosen Test Points')
standardFlatLimitsAndLabels(plt)
plt.scatter(sx, sy,  marker='o',  color='lightblue')
plt.scatter(X_test[:,0], X_test[:,1], marker='1', color='black')
plt.show()

# Plot the projection 

# Training Data Projection 1
plt.suptitle('Projection of Predictions vs X of Full Dataset and Test Points')
plt.scatter(sx, sz,  marker='o', color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('X')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_test[:,0], y_test_pred,  marker='1', color='blue')
plt.show()

# Training Data Projection 2
plt.suptitle('Projection of Predictions vs Y of Full Dataset and Test Points')
plt.scatter(sy, sz,  marker='o',  color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('Y')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_test[:,1], y_test_pred,  marker='1',  color='blue')
plt.show()









