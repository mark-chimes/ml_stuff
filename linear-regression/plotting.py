# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:23:04 2021

@author: mark.chimes
"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random

seed = 444
random.seed(seed)

mu, sigma = 0, 0.1 # mean and standard deviation
sx = np.random.normal(mu, sigma, 1000)
sy = np.random.normal(mu, sigma, 1000)
sz_diff = np.random.normal(mu, sigma, 1000)
sz = -(sx+sy) + sz_diff

plt.scatter(sx, sy, color='black')
plt.show()

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
# Plot the surface.
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
ax.scatter3D(sx, sy, -2, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')
ax.scatter3D(sx, -1, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')
ax.scatter3D(-1, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.08, cmap='hot')

plt.show()

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, -45)
# Plot the surface.
ax.scatter3D(sx, sy, sz, c = -np.sqrt(sx**2 + sy**2), alpha=0.8, cmap='hot')
plt.show()



N = 20
d = 0.2

#X = [random.uniform(0, 1) for x in range(N)]
#X.sort
#Y = [x + random.uniform(-d, d) for x in X]
X = [-3,-2,-1,0,1,2,3]
Y = [-3,-2,-1,0,1,2,3]
A,B = np.meshgrid(X, Y)
Z = np.absolute(A)+np.absolute(B)

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
# Plot the surface.
surf = ax.plot_surface(A, B, Z, cmap='viridis',
                       linewidth=0, antialiased=False)
plt.show()









plt.scatter(A, B, color='blue')
plt.scatter(X, [0 for x in X], color='black')
plt.scatter([0 for y in Y], Y, color='black')
plt.show()


plt.scatter(X, Y, color='black')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter3D(X, Y,  cmap='viridis')
ax.view_init(20, 20)
plt.show()

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, 0)
ax.scatter3D(X, Y,  cmap='viridis')
plt.show()

#Z = [z for z in range(N)]
Z = [x for x in X]

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(20, 20)
ax.scatter3D(X, Y, Z,  cmap='viridis')
plt.show()

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(0, 0)
ax.scatter3D(X, Y, Z,  cmap='viridis')
plt.show()

ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(90, 0)
ax.scatter3D(X, Y, Z,  cmap='viridis')
plt.show()


