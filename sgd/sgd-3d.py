# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:17:24 2021

@author: mark.chimes
"""

#%% Imports

import math 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


#%% Display Limits
x_min, x_max = -1, 1 
y_min, y_max = -1, 1

x_min_zoom, x_max_zoom = -0.5, 0.5 
y_min_zoom, y_max_zoom = -0.5, 0.5

xlim = [x_min, x_max]
ylim = [y_min, y_max]

xlim_zoom = [x_min_zoom, x_max_zoom]
ylim_zoom = [y_min_zoom, y_max_zoom]

axlim = [-1, 1]
aylim = [-1, 1]
azlim = [-1, 1]

#%% Generate Flat Data

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


#%% Plotting Functions

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
  

def standardFlatLimitsAndLabels(plt): 
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x')
    h = plt.ylabel('y')
    h.set_rotation(0)    


#%% Plot Flat Data

plt.suptitle('Base Data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sax, say, marker='.', color='red')
plt.scatter(sbx, sby, marker='.', color='blue')
plt.show()


#%% Generate floating data

def generatingFunction(x, y): 
    return 0.5*x + x**2 + 0.5*y + y**2

gx = gy = np.arange(-0.75, 0.75, 0.1)
gX, gY = np.meshgrid(gx, gy)
gZ0 = np.zeros(gX.shape)
gz = np.array(np.ravel(generatingFunction(gX, gY)))
gZ = gz.reshape(gX.shape)

def predictionWireframeFloatAndPredictions(z, main_alph=0.2, proj_alph=0.05, v_angle=0, h_angle=0):
    Z = -np.sqrt(gX**2 + gY**2)
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    
    def plot_surfy(ix, iy, iz, ialph): 
         ax.plot_surface(ix, iy, iz, facecolors=colors, rcount=rcount, ccount=ccount, alpha=ialph)
         
    plot_surfy(gX, gY, z, 0.2)
    plot_surfy(gX, gY, gZ0+z_reflect(v_angle), 0.05)
    plot_surfy(gX, y_reflect(h_angle), z, 0.05)
    plot_surfy(x_reflect(h_angle), gY, z, 0.05)

#%% Generate floating data

saz = generatingFunction(sax, say)
sbz = generatingFunction(sbx, sby)


def floatAndProjections(x,y,z, col='green', main_alph=0.8, proj_alph=0.08, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z,  marker='.', color=col, alpha=main_alph)
    ax.scatter3D(x, y, z_reflect(v_angle),  color=col, alpha=proj_alph)
    ax.scatter3D(x, y_reflect(h_angle), z,  color=col, alpha=proj_alph)
    ax.scatter3D(x_reflect(h_angle), y, z,  color=col, alpha=proj_alph)

for a, b in angles: 
    # Plot the points with true surface.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset and \n' + r'Generating Function Wireframe')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    # floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(gZ, 0.05, 0.01, a, b)
    floatAndProjections(sax, say, saz, col='red', main_alph=0.5, proj_alph=0.01)
    floatAndProjections(sbx, sby, sbz, col='blue', main_alph=0.5, proj_alph=0.002)
    plt.show()
    
#%% Stochastic Gradient Descent


total = Na + Nb
test_N = 100
train_N = total - test_N

SX = np.concatenate((sax, sbx))
SY = np.concatenate((say, sby))
SZ = generatingFunction(sx, sy)

SC = np.concatenate((np.ones(sax.shape), np.zeros(sbx.shape)))

all_points = np.column_stack((SX, SY, SZ, SC))

np.random.shuffle(all_points)









