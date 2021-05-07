# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets, linear_model
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

#%%  Least Squares

col1 = np.ones(N)
desmat = np.column_stack((col1, sx, sy))
sol, res, rnk, sing = lstsq(desmat, sz)




#%% Wireframe
gx = gy = np.arange(-0.75, 0.75, 0.1)
gX, gY = np.meshgrid(gx, gy)
gZ0 = np.zeros(gX.shape)

# gz = np.array(-np.ravel(gX) - np.ravel(gY))
# gZ = gz.reshape(gX.shape)

gz_sol_a =  np.array(sol[1]*np.ravel(gX) + sol[2]*np.ravel(gY))
gZ_sol = sol[0]*np.ones(gX.shape) + gz_sol_a.reshape(gX.shape)





#%% PLOTS
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

#%% Generating

# Plot the points with true generating function.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.3, cmap='hot')
ax.scatter3D(sx, sy, -2, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.scatter3D(sx, -1, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.scatter3D(-1, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.scatter3D(sx, sy, target_sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.3, cmap='Greens')
ax.scatter3D(sx, -1, target_sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='Greens')
ax.scatter3D(-1, sy, target_sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='Greens')
plt.show()

# Plot the side-on view.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, -45)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
ax.scatter3D(sx, sy, target_sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.5, cmap='Greens')
plt.show()

#%% Generating Surface

# Plot the points with true surface.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.3, cmap='hot')
ax.scatter3D(sx, sy, -2, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.scatter3D(sx, -1, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.scatter3D(-1, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.03, cmap='hot')
ax.plot_wireframe(gX, gY, gZ, alpha=0.2)
ax.plot_wireframe(gX, gY, gZ0-2, alpha=0.1)
ax.plot_wireframe(gX, -1, gZ, alpha=0.1)
ax.plot_wireframe(-1, gY, gZ, alpha=0.1)
plt.show()

# Plot the side-on view.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, -45)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
ax.plot_wireframe(gX, gY, gZ, alpha=0.8)
plt.show()

#%% Approximating

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
ax.plot_wireframe(gX, gY, gZ_sol, alpha=0.2)
ax.plot_wireframe(gX, gY, gZ0-2, alpha=0.05)
ax.plot_wireframe(gX, -1, gZ_sol, alpha=0.05)
ax.plot_wireframe(-1, gY, gZ_sol, alpha=0.05)
plt.show()

# Plot the side-on view.
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, -45)
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
ax.plot_wireframe(gX, gY, gZ_sol, alpha=0.8)
plt.show()























