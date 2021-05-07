# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import lstsq
import random

seed = 444
random.seed(seed)
N = 1000

#%% Generate Data
mu, sigma = 0, 0.1 # mean and standard deviation
sx = np.random.normal(mu, sigma, N)
sy = np.random.normal(mu, sigma, N)
sz_diff = np.random.normal(mu, sigma, N)
sz = -(sx+sy) + sz_diff
target_sz = -1*sx + -1*sy

#%% Flat Plots
# Plot the normal distribution
plt.scatter(sx, sy, color='black')
plt.show()

# Plot the points.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
ax.scatter3D(sx, sy, -2, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')
ax.scatter3D(sx, -1, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')
ax.scatter3D(-1, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')
plt.show()

# Plot the side-on view.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, -45)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
plt.show()


#%% Wireframe setup
gx = gy = np.arange(-0.75, 0.75, 0.1)
gX, gY = np.meshgrid(gx, gy)
gZ0 = np.zeros(gX.shape)
gz = np.array(-np.ravel(gX) - np.ravel(gY))
gZ = gz.reshape(gX.shape)

#%% Close all plots
plt.close('all')


#%% Linear Regression Loop

training_nums = [1,2,3,4,5,6,7,8,9,10,15,20,30,50,80,100,200]


for num_train in training_nums:
    
    # Linear Regression
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
    
    XX = np.column_stack((gX.reshape(gX.shape[0]*gX.shape[1],), gY.reshape(gY.shape[0]*gY.shape[1])))
    yy = regr.predict(XX).reshape(gX.shape)
    
    

    #  Linear Regression Surface
    
    # Plot the points with approximated surface.
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(20, 20)
    ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.3, cmap='hot')
    ax.scatter3D(sx, sy, -2, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
    ax.scatter3D(sx, -1, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
    ax.scatter3D(-1, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
    ax.plot_wireframe(gX, gY, yy, alpha=0.2)
    ax.plot_wireframe(gX, gY, gZ0-2, alpha=0.05)
    ax.plot_wireframe(gX, -1, yy, alpha=0.05)
    ax.plot_wireframe(-1, gY, yy, alpha=0.05)
    plt.show()




#%% Do not execute

func do_nothing: 
    # Training and Test Data Plots

    plt.scatter(X_train[:,0], X_train[:,1], color='black')
    plt.show()
    #### 
        
    # Plot the side-on view.
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(0, -45)
    ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
    ax.plot_wireframe(gX, gY, yy, alpha=0.8)
    plt.show()
    
    
    # Linear Regression Surface on Training Data
    
    tx = X_train[:,0]
    ty = X_train[:,1]
    tz = y_train[:]
    
    # Plot the points with approximated surface.
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(20, 20)
    ax.scatter3D(tx, ty, tz, c = -np.sqrt(tx**2 + ty**2), alpha=0.3, cmap='hot')
    ax.scatter3D(tx, ty, -2, c = -np.sqrt(tx**2 + ty**2), alpha=0.03, cmap='hot')
    ax.scatter3D(tx, -1, tz, c = -np.sqrt(tx**2 + ty**2), alpha=0.03, cmap='hot')
    ax.scatter3D(-1, ty, tz, c = -np.sqrt(tx**2 + ty**2), alpha=0.03, cmap='hot')
    ax.plot_wireframe(gX, gY, yy, alpha=0.2)
    ax.plot_wireframe(gX, gY, gZ0-2, alpha=0.05)
    ax.plot_wireframe(gX, -1, yy, alpha=0.05)
    ax.plot_wireframe(-1, gY, yy, alpha=0.05)
    plt.show()
    
    # Plot the side-on view.
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(0, -45)
    ax.scatter3D(tx, ty, tz, c = -np.sqrt(tx**2 + ty**2), alpha=0.8, cmap='hot')
    ax.plot_wireframe(gX, gY, yy, alpha=0.8)
    plt.show()
    




