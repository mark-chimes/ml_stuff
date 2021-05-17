# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import cm
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

seed = 444
random.seed(seed)

limi = 5

xlim = [-limi, limi]
ylim = [-limi, limi]
axlim = [-limi, limi]
aylim = [-limi, limi]
azlim = [-limi, limi]


#%% Define Plotting Functions 

angles = [(45,45), (45,20), (45,0), (20,0), (0,0), \
          (0,-20), (0,-45), (20,-45), (45,-45)]
    
def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X')
    h = plt.ylabel('Y')
    h.set_rotation(0)    

def standardAxLimitsAndLabels(ax): 
    ax.set_xlim(axlim)
    ax.set_ylim(aylim)
    ax.set_zlim(azlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
def x_reflect(angle): 
    if 90 < angle and angle <= 270:
        return 1
    if -270 <= angle and angle < -90:
        return 1
        return -1
    return -1

def y_reflect(angle): 
    if 0 <= angle and angle < 180:
        return -1
    if -360 <= angle and angle < -180:
        return -1
    return 1

def z_reflect(angle): 
    if 0 <= angle and angle < 180:
        return -1
    if -360 <= angle and angle < -180:
        return -1
    return 1

def floatAndProjections(x,y,z, col, main_alph=0.8, proj_alph=0.08, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z,  marker='o', c =  col, alpha=main_alph)
    ax.scatter3D(x, y, limi*z_reflect(v_angle), c=col, alpha=proj_alph)
    ax.scatter3D(x, limi*y_reflect(h_angle), z, c=col, alpha=proj_alph)
    ax.scatter3D(limi*x_reflect(h_angle), y, z, c=col, alpha=proj_alph)


#%% 3 Features
X1, Y1 = make_classification(n_features=3, n_redundant=0, n_informative=3,
                             n_clusters_per_class=1)

# Flat Plots
plt.suptitle('Input data')
standardFlatLimitsAndLabels(plt)

plt.title("Three informative features, two classes", fontsize='small')
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 0], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()

# 3D Plot

for a, b in angles: 
    # Plot the points.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset (with shadows / projections)')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a,b)
    floatAndProjections(X1[:, 0], X1[:, 1],X1[:, 2], Y1, v_angle=a, h_angle=b)
    plt.show()   

#%% 3 Features 3 classes
X1, Y1 = make_classification(n_features=3, n_classes=3, n_redundant=0, n_informative=3,
                             n_clusters_per_class=1)

# Flat Plots
plt.suptitle('Input data')
standardFlatLimitsAndLabels(plt)

plt.title("Three informative features, three classes", fontsize='small')
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 0], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()

# 3D Plot

for a, b in angles: 
    # Plot the points.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset (with shadows / projections)')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a,b)
    floatAndProjections(X1[:, 0], X1[:, 1],X1[:, 2], Y1, v_angle=a, h_angle=b)
    plt.show()   


#%% 3 Features 3 ckasses 1 redundant feature
X1, Y1 = make_classification(n_features=3, n_classes=3, n_redundant=1, n_informative=2,
                             n_clusters_per_class=1)

# Flat Plots
plt.suptitle('Input data')
standardFlatLimitsAndLabels(plt)

plt.title("Two informative features, one redundant", fontsize='small')
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 0], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()
plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()

# 3D Plot

for a, b in angles: 
    # Plot the points.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset (with shadows / projections)')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a,b)
    floatAndProjections(X1[:, 0], X1[:, 1],X1[:, 2], Y1, v_angle=a, h_angle=b)
    plt.show()   
