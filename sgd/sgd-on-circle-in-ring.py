# -*- coding: utf-8 -*-
"""
Created on Tue MaZ 11 14:43:06 2021

@author: mark.chimes
"""

#%% Imports
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from matplotlib.colors import ListedColormap

#%% Random Seed
#import random
#seed = 444
#random.seed(seed)

#%% Display Limits
x_min, x_max = -1, 1 
y_min, y_max = -1, 1

x_min_zoom, x_max_zoom = -0.5, 0.5 
y_min_zoom, y_max_zoom = -0.5, 0.5

#%% Generate Data
# Two sets of data: a and b. Horizontal x vertical y

# Ring
Na = 1000
siga = 0.1
ellipse_width = 0.7
ellipse_height = 0.7
thetas = np.linspace(0, 2*math.pi, Na)

#generate deviations
devs = np.random.normal(0, siga, Na)
x_mults = ellipse_width*(np.ones(Na)+devs)
y_mults = ellipse_width*(np.ones(Na)+devs)

sax = x_mults*np.cos(thetas)
say = y_mults*np.sin(thetas)

# Disc
Nb = 1000
sigb = 0.1
mu_bx, mu_by, sigma_bx, sigma_by = 0, 0, sigb, sigb # mean and standard deviation
sbx = np.random.normal(mu_bx, sigma_bx, Nb)
sby = np.random.normal(mu_by, sigma_by, Nb)

#%% Plot Base Data

xlim = [x_min, x_max]
ylim = [y_min, y_max]

xlim_zoom = [x_min_zoom, x_max_zoom]
ylim_zoom = [y_min_zoom, y_max_zoom]

# xlim = [-1, 1]
# ylim = [-1, 1]

def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x')
    h = plt.ylabel('y')
    h.set_rotation(0)    

plt.suptitle('Base Data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sax, say, marker='.', color='blue')
plt.scatter(sbx, sby, marker='.', color='red')
plt.show()

#%% Data in correct format and shuffled

total = Na + Nb
test_N = 100
train_N = total - test_N

X = np.concatenate((sax, sbx))
Y = np.concatenate((say, sby))
Z_categorize = np.concatenate((np.zeros(Na), np.ones(Nb))) # class
all_points = np.column_stack((X, Y, Z_categorize))
np.random.shuffle(all_points)
XY = all_points[:,:-1] # first two columns
Z = np.ravel(all_points[:,-1:]) # last column
Z_colors = np.array(['blue' if c > 0.5 else 'red' for c in Z])

plt.suptitle('Plotting After Shuffle using Z-values for color')
standardFlatLimitsAndLabels(plt)
plt.scatter(XY[:,:1], XY[:,1:], marker='.', c=Z_colors)
plt.show()

XY_train  = XY[:train_N,:]
XY_test  = XY[train_N:,:]

Z_train = Z[:train_N]
Z_test = Z[train_N:]

Z_train_colors = np.array(['blue' if x > 0.5 else 'red' for x in Z_train]).T
Z_test_colors = np.array(['cyan' if x > 0.5 else 'pink' for x in Z_test]).T

plt.suptitle('Training and Test Datasets')
standardFlatLimitsAndLabels(plt)
plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='2', c=Z_train_colors)
plt.scatter(XY_test[:,:1], XY_test[:,1:], marker=',', c=Z_test_colors)

plt.show()

#%% Stochastic Gradient Descent
for iterations in [10,10,10,10,10]:
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=iterations)
    clf.fit(XY_train, Z_train)
    
    Z_predict = clf.predict(XY_test)
    
    cmap_light = ListedColormap(['pink', 'cyan'])
    
    h = .02  # step size in the mesh
    xx, xy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z_mesh = clf.predict(np.c_[xx.ravel(), xy.ravel()])
    
    Z_mesh_reshape = Z_mesh.reshape(xx.shape)
    #plt.figure(figsize=(8, 6))
    
    standardFlatLimitsAndLabels(plt)
    title = 'Max iterations: ' + str(iterations)
    plt.suptitle(title)
    plt.contourf(xx, xy, Z_mesh_reshape, cmap=cmap_light)
    plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='.', c=Z_train_colors)
    plt.scatter(XY_test[:,:1], XY_test[:,1:], marker='.', c=Z_predict)

    plt.show()














