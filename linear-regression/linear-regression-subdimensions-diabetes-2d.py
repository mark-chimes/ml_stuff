# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from scipy.linalg import lstsq
import random
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm
from matplotlib.colors import ListedColormap

seed = 444
random.seed(seed)

#xlim = [-1, 1]
#ylim = [-1, 1]

#%% Generate Data
N = 442
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

#%% Linear Regression
num_train = 400

X = diabetes_X
X_train = X[:num_train,:]
X_test = X[num_train:,:]

y_train = diabetes_y[:num_train]
y_test = diabetes_y[num_train:]

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


#%% Define Plotting Functions 
   
def standardFlatLimitsAndLabels(plt, i): 
    plt.scatter(X[:,i], diabetes_y,  marker='o', color='orange')
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X[' + str(i) + ']')

#%%  Training Data Projection 
for i in range(0,10):
    plt.suptitle('Projection of Outputs vs X['+ str(i) + '] of Full Dataset and Training Points')
    standardFlatLimitsAndLabels(plt, i)
    plt.scatter(X_train[:,i], y_train_pred,  marker='1', color='green')
    plt.show()

#%% Test Data Projection
for i in range(0,10):
    plt.suptitle('Projection of Predictions vs X[0] of Full Dataset and Test Points')
    standardFlatLimitsAndLabels(plt, i)
    plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.show()

    
#%% True Data Colormap
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('True data all')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X[:,i], diabetes_y,  marker='o', c=diabetes_y)
    plt.show()


#%% Training Data Colormap
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('Predicted data all')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X_train[:,i], y_train,  marker='o', c=y_train_pred)
    plt.show()

#%% Test Data Colormap
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X_test')
    plt.suptitle('Projection of Predictions vs X[0] of Test Dataset and Test Points')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X_test[:,i], y_test,  marker='o', c=y_test_pred)
    plt.show()


#%% True Data Colormap
cmap_light = ListedColormap(['pink', 'cyan'])
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('True data all')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X[:,i], diabetes_y,  marker='o', cmap=cmap_light, c=diabetes_y)
    plt.show()


#%% Training Data Colormap
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('Predicted data all')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X_train[:,i], y_train,  marker='o',cmap=cmap_light, c=y_train_pred)
    plt.show()

#%% Test Data Colormap
for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X_test')
    plt.suptitle('Projection of Predictions vs X[0] of Test Dataset and Test Points')
    # plt.scatter(X_test[:,i], y_test_pred,  marker='1', color='blue')
    plt.scatter(X_test[:,i], y_test,  marker='o',cmap=cmap_light, c=y_test_pred)
    plt.show()

#%% Yes / No Data Colormap
cmap_predict = ListedColormap(['red', 'green'])
class_thing = np.array([1 if y > 200 else -1 for y in diabetes_y])
class_thing_all_prediction = np.array([1 if y > 200 else -1 for y in y_all_pred])
class_thing_color = class_thing*class_thing_all_prediction

for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('Projection of Predictions vs X of Test Dataset and Test Points')
    plt.scatter(diabetes_X[:,i], y_all_pred,  marker='o',cmap=cmap_predict, c=class_thing)
    plt.show()

#%% Distance Colormap
distance_colors = abs(y_all_pred - diabetes_y)

for i in range(0,10):
    h = plt.ylabel('y')
    h.set_rotation(0)
    plt.xlabel('X')
    plt.suptitle('Projection of Predictions vs X of Test Dataset and Test Points')
    plt.scatter(diabetes_X[:,i], y_all_pred,  marker='o',cmap=cm.magma, c=distance_colors)
    plt.show()

#%% Yes / No Data Colormap Combinations
cmap_predict = ListedColormap(['red', 'green'])
class_thing = np.array([1 if y > 200 else -1 for y in diabetes_y])
class_thing_all_prediction = np.array([1 if y > 200 else -1 for y in y_all_pred])
class_thing_color = class_thing*class_thing_all_prediction

for i in range(0,10):
    for j in range(i+1,10):
        h = plt.ylabel('X[' + str(j) + ']')
        h.set_rotation(0)
        plt.xlabel('X[' + str(i) + ']')
        plt.suptitle('Projection of X vs X of Test Dataset and Test Points')
        plt.scatter(diabetes_X[:,i], diabetes_X[:,j],  marker='o',cmap=cmap_predict, c=class_thing)
        plt.show()


#%% Distance Colormap Combinations
distance_colors = abs(y_all_pred - diabetes_y)

for i in range(0,10):
    for j in range(i+1,10):
        h = plt.ylabel('X[' + str(j) + ']')
        h.set_rotation(0)
        plt.xlabel('X[' + str(i) + ']')
        plt.suptitle('Projection of X vs X of Test Dataset and Test Points')
        plt.scatter(diabetes_X[:,i], diabetes_X[:,j],  marker='o',cmap=cm.magma, c=distance_colors)
        plt.show()
