# -*- coding: utf-8 -*-
"""
Created on Tue MaZ 11 14:43:06 2021

@author: mark.chimes
"""

#%% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

#%% Random Seed
#import random
#seed = 444
#random.seed(seed)

#%% DisplaZ Limits
x_min, x_max = -1, 1 
y_min, y_max = -1, 1

x_min_zoom, x_max_zoom = -0.5, 0.5 
y_min_zoom, y_max_zoom = -0.5, 0.5

#%% Generate Data
# Two sets of data: a and b. Horizontal x vertical y

lin_dist = 0.2 # move the two data sets apart
sig = 0.1

# datapoints per group
Na = 1000
Nb = 1000

mu_ax, mu_ay, sigma_ax, sigma_ay  = -lin_dist, -lin_dist, sig, sig*2 # mean and standard deviation
mu_bx, mu_by, sigma_bx, sigma_by  = lin_dist, lin_dist, sig, sig*2 # mean and standard deviation

sax = np.random.normal(mu_ax, sigma_ax, Na)
say = np.random.normal(mu_ay, sigma_ay, Na)
sbx = np.random.normal(mu_bx, sigma_bx, Na)
sby = np.random.normal(mu_by, sigma_by, Na)

# TODO Regularize Data

#%% Plot Base Data

xlim = [x_min, x_max]
ylim = [y_min, y_max]

xlim_zoom = [x_min_zoom, x_max_zoom]
ylim_zoom = [y_min_zoom, y_max_zoom]

# xlim = [-1, 1]
# ylim = [-1, 1]

def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim_zoom)
    plt.ylim(ylim_zoom)
    plt.xlabel('x')
    h = plt.ylabel('y')
    h.set_rotation(0)    

plt.suptitle('Base Data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sax, say, marker='.', color='red')
plt.scatter(sbx, sby, marker='.', color='blue')
plt.show()

#%% Data in correct format and shuffled

total = Na + Nb
test_N = 200
train_N = total - test_N

X = np.concatenate((sax, sbx))
Y = np.concatenate((say, sby))
Z_categorize = np.concatenate((np.zeros(Na), np.ones(Nb))) # class
all_points = np.column_stack((X, Y, Z_categorize))
np.random.shuffle(all_points)
X = all_points[:,:-1] # first two columns
Z = np.ravel(all_points[:,-1:]) # last column
Z_colors = np.array(['blue' if c > 0.5 else 'red' for c in Z])

plt.suptitle('Plotting After Shuffle using Z-values for color')
standardFlatLimitsAndLabels(plt)
plt.scatter(X[:,:1], X[:,1:], marker='.', c=Z_colors)
plt.show()

X_train  = X[:train_N,:]
X_test  = X[train_N:,:]

Z_train = Z[:train_N]
Z_test = Z[train_N:]

Z_train_colors = np.array(['blue' if x > 0.5 else 'red' for x in Z_train]).T
Z_test_colors = np.array(['cyan' if x > 0.5 else 'pink' for x in Z_test]).T

plt.suptitle('Training and Test Datasets')
standardFlatLimitsAndLabels(plt)
plt.scatter(X_test[:,:1], X_test[:,1:], marker=',', c=Z_test_colors)
plt.scatter(X_train[:,:1], X_train[:,1:], marker='2', c=Z_train_colors)

plt.show()

#%% Plot hyperplane

def plot_hyperplane(coef, intercept):
    coef0 = coef[0,0]
    coef1 = coef[0,1]
    
    icpt = intercept[0]
    
    m = coef0 / -coef1
    c = icpt / -coef1

    def line(x0):
        return m*x0 + c

    plt.plot([x_min, x_max], [line(x_min), line(x_max)], \
             ls="--", color='black')



#%% Stochastic Gradient Descent

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train, Z_train)

coef = clf.coef_
intercept = clf.intercept_


Z_predict = clf.predict(X_test)

cmap_light = ListedColormap(['pink', 'cyan'])

h = .02  # step size in the mesh
xx, xy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z_mesh = clf.predict(np.c_[xx.ravel(), xy.ravel()])

Z_mesh_reshape = Z_mesh.reshape(xx.shape)
#plt.figure(figsize=(8, 6))

standardFlatLimitsAndLabels(plt)
title = 'Perceptron Classification'
plt.suptitle(title)
plt.contourf(xx, xy, Z_mesh_reshape, cmap=cmap_light)
plt.scatter(X_train[:,:1], X_train[:,1:], marker='.', c=Z_train_colors)

plot_hyperplane(coef, intercept)

plt.show()








