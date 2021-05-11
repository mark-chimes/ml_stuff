# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:28:01 2021

@author: mark.chimes
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn import neighbors

#%% Random
seed = 444
random.seed(seed) # Apparently you're not really supposed to do this...?

#%% Display Limits
x_min, x_max = -1, 1 
y_min, y_max = -1, 1

x_min_zoom, x_max_zoom = -0.5, 0.5 
y_min_zoom, y_max_zoom = -0.5, 0.5

xlim = [x_min, x_max]
ylim = [y_min, y_max]

xlim_zoom = [x_min_zoom, x_max_zoom]
ylim_zoom = [y_min_zoom, y_max_zoom]

#%% Generate Data

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

def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
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

plt.suptitle('Plotting After Shuffle using y-values for color')
standardFlatLimitsAndLabels(plt)
plt.scatter(XY[:,:1], XY[:,1:], marker='.', c=Z_colors)
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
    clf.fit(XY, Z)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    contour_Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    cmap_light = ListedColormap(['pink', 'cyan'])
    
    # Put the result into a color plot
    contour_Z = contour_Z.reshape(xx.shape)
    
    title = 'Nearest Neighbour Classification for k = ' + str(n_neighbors)
    plt.suptitle(title)
    standardFlatLimitsAndLabels(plt)
    #plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, contour_Z, cmap=cmap_light)
    plt.scatter(XY[:,:1], XY[:,1:], marker='.', c=Z_colors)
    plt.show()
    





















































