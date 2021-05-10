# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:28:01 2021

@author: mark.chimes
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn import neighbors


### TODO I really should figure out a consistent variable naming scheme
### This X, y, x1, x2, xx, yy, Z nonsense is annoying

#%% Generate Data

seed = 444
random.seed(seed) # Apparently you're not really supposed to do this...?

x_min, x_max = -0.5, 0.5 
y_min, y_max = -0.5, 0.5 


lin_dist = 0.3
sig = 0.1

N = 1000 # datapoints per group

mu1_x, mu1_y, sigma1_x, sigma1_y  = -lin_dist, -lin_dist, sig, sig*2 # mean and standard deviation

s1x = np.random.normal(mu1_x, sigma1_x, N)
s1y = np.random.normal(mu1_y, sigma1_y, N)

mu2_x, mu2_y, sigma2_x, sigma2_y = lin_dist, lin_dist, sig*2, sig # mean and standard deviation
s2x = np.random.normal(mu2_x, sigma2_x, N)
s2y = np.random.normal(mu2_y, sigma2_y, N)

#%% Plot Base Data

xlim = [x_min, x_max]
ylim = [y_min, y_max]

def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x1')
    h = plt.ylabel('x2')
    h.set_rotation(0)    

plt.suptitle('Base Data')
standardFlatLimitsAndLabels(plt)
plt.scatter(s1x, s1y, marker='.', color='red')
plt.scatter(s2x, s2y, marker='.', color='blue')
plt.show()

#%% Data in correct format and shuffled

total = 2*N
test_N = 100
train_N = total - test_N

X1 = np.concatenate((s1x, s2x))
X2 = np.concatenate((s1y, s2y))
y_categorize = np.concatenate((np.zeros(N), np.ones(N))) # class
all_points = np.column_stack((X1, X2, y_categorize))
np.random.shuffle(all_points)
X = all_points[:,:-1] # first two columns
y = np.ravel(all_points[:,-1:]) # last column
y_colors = np.array(['blue' if x > 0.5 else 'red' for x in y])

plt.suptitle('Plotting After Shuffle using y-values for color')
standardFlatLimitsAndLabels(plt)
plt.scatter(X[:,:1], X[:,1:], marker='.', c=y_colors)
plt.show()

'''
X_train  = X[:train_N,:]
X_test  = X[train_N:,:]

y_train = y[:train_N]
y_test = y[train_N:]

y_train_colors = np.array(['blue' if x > 0.5 else 'red' for x in y_train]).T
y_test_colors = np.array(['blue' if x > 0.5 else 'red' for x in y_test]).T

plt.suptitle('Training dataset')
standardFlatLimitsAndLabels(plt)
plt.scatter(X_train[:,:1], X_train[:,1:], marker='2', c=y_train_colors)
plt.scatter(X_test[:,:1], X_test[:,1:], marker=',', c=y_test_colors)

plt.show()
'''

#%% Create mesh

h = .02  # step size in the mesh

xx, yy = np.meshgrid(np.arange(x_min-0.1, x_max+0.1, h),
                 np.arange(y_min-0.1, y_max+0.1, h))
#%% Nearest Neighbours

for n_neighbors in [1,2,5,10,15,20]:
    # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    cmap_light = ListedColormap(['pink', 'cyan'])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    title = 'Nearest Neighbour Classification for k = ' + str(n_neighbors)
    plt.suptitle(title)
    standardFlatLimitsAndLabels(plt)
    #plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:,:1], X[:,1:], marker='.', c=y_colors)
    plt.show()
    
    




















