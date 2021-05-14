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
from sklearn.linear_model import SGDClassifier


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

#angles = [(45,45), (45,20), (45,0), (20,0), (0,0), \
#          (0,-20), (0,-45), (20,-45), (45,-45)]

angles = [(45,45), (20,0)]

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


def predictionWireframeFloatAndPredictions(gen_func, main_alph=0.2, proj_alph=0.05, v_angle=0, h_angle=0):
    gx = gy = np.arange(-0.75, 0.75, 0.1)
    gX, gY = np.meshgrid(gx, gy)
    gZ0 = np.zeros(gX.shape)

    Z = -np.sqrt(gX**2 + gY**2)
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    
    def plot_surfy(ix, iy, iz, ialph): 
         ax.plot_surface(ix, iy, iz, facecolors=colors, rcount=rcount, ccount=ccount, alpha=ialph)
         
    plot_surfy(gX, gY, gen_func(gX, gY), 0.2)
    plot_surfy(gX, gY, gZ0+z_reflect(v_angle), 0.05)
    plot_surfy(gX, y_reflect(h_angle), gen_func(gX, gY), 0.05)
    plot_surfy(x_reflect(h_angle), gY, gen_func(gX, gY), 0.05)

def floatAndProjections(x,y,z, col='green', main_alph=0.8, proj_alph=0.08, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z,  marker='.', color=col, alpha=main_alph)
    ax.scatter3D(x, y, z_reflect(v_angle),  color=col, alpha=proj_alph)
    ax.scatter3D(x, y_reflect(h_angle), z,  color=col, alpha=proj_alph)
    ax.scatter3D(x_reflect(h_angle), y, z,  color=col, alpha=proj_alph)


#%% Plot Flat Data

plt.suptitle('Base Data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sax, say, marker='.', color='blue')
plt.scatter(sbx, sby, marker='.', color='red')
plt.show()


#%% Generating function

def generatingFunction(x, y): 
    return 0.5*x + x**2 + 0.5*y + y**2

#%% Plot points and the generating function wireframe

saz = generatingFunction(sax, say)
sbz = generatingFunction(sbx, sby)

for a, b in angles: 
    # Plot the points with true surface.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset and \n' + r'Embedding Wireframe')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    # floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(generatingFunction, 0.05, 0.01, a, b)
    floatAndProjections(sax, say, saz, col='blue', main_alph=0.5, proj_alph=0.01)
    floatAndProjections(sbx, sby, sbz, col='red', main_alph=0.5, proj_alph=0.01)
    plt.show()
    
#%% Stochastic Gradient Descent


total = Na + Nb
test_N = 100
train_N = total - test_N

SX = np.concatenate((sax, sbx))
SY = np.concatenate((say, sby))
SZ = generatingFunction(SX, SY)

SC = np.concatenate((np.ones(sax.shape), np.zeros(sbx.shape)))

all_points = np.column_stack((SX, SY, SZ, SC))
np.random.shuffle(all_points)
X = all_points[:,:-1] # first two columns
y = np.ravel(all_points[:,-1:]) # last column
y_colors = np.array(['blue' if c > 0.5 else 'red' for c in y])

for a, b in angles: 
    # Plot the points with true surface.
    ax = plt.axes(projection='3d')
    plt.suptitle('Plotting After Shuffle using Z-values for color')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    # floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(generatingFunction, 0.05, 0.01, a, b)
    floatAndProjections(X[:,0:1], X[:,1:2], X[:,2:3], col=y_colors, main_alph=0.5, proj_alph=0.01)
    plt.show()
    

X_train  = X[:train_N,:]
X_test  = X[train_N:,:]

y_train = y[:train_N]
y_test = y[train_N:]

y_train_colors = np.array(['blue' if x > 0.5 else 'red' for x in y_train]).T
y_test_colors = np.array(['cyan' if x > 0.5 else 'pink' for x in y_test]).T

plt.suptitle('Training and Test Datasets')
standardFlatLimitsAndLabels(plt)
plt.scatter(X_train[:,0:1], X_train[:,1:2], marker='2', c=y_train_colors)
plt.scatter(X_test[:,0:1], X_test[:,1:2], marker=',', c=y_test_colors)
plt.show()

for iterations in [10]:
    #iterations = 10
    
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=iterations)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    y_predict_colors = np.array(['blue' if x > 0.5 else 'red' for x in y_predict]).T
    
    coef = clf.coef_
    intercept = clf.intercept_
    coef0 = coef[0,0]
    coef1 = coef[0,1]
    coef2 = coef[0,2]
    icpt = intercept[0]
    
    m1 = coef0 / -coef2
    m2 = coef1 / -coef2
    c = icpt / -coef2
    
    plane_func = lambda x,y :  m1*x + m2*y + c
    
    for a, b in angles: 
        
        # Plot the points with true surface.
        ax = plt.axes(projection='3d')
        plt.suptitle('Predicted values after ' + str(iterations) + ' iterations with separating plane')
        standardAxLimitsAndLabels(ax)
        ax.view_init(a, b)
        # floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
        # predictionWireframeFloatAndPredictions(generatingFunction, 0.05, 0.01, a, b)
    
        predictionWireframeFloatAndPredictions(plane_func, v_angle=a, h_angle=b)
    
        # ax.plot_surface(XG, YG, ZG, color='green', alpha=0.25)
        
        floatAndProjections(X_test[:,0:1], X_test[:,1:2], X_test[:,2:3], col=y_predict_colors, main_alph=0.5, proj_alph=0.01)
        plt.show()

    gx = gy = np.arange(-1, 1, 0.01)
    gX, gY = np.meshgrid(gx, gy)
    gZ = np.column_stack((gX.ravel(), gY.ravel(), generatingFunction(gX, gY).ravel()))
    y_predict = clf.predict(gZ)
    
    plt.suptitle('Predicted Values')
    standardFlatLimitsAndLabels(plt)
    Z_predict_colors = np.array(['cyan' if x > 0.5 else 'magenta' for x in y_predict]).T

    plt.scatter(gX, gY, marker='.', c=Z_predict_colors)
    plt.scatter(sax, say, marker='.', color='blue')
    plt.scatter(sbx, sby, marker='.', color='red')
    plt.show()
'''
for iterations in [20]: # [1,2,5,10, 20]:
    ax = plt.axes(projection='3d')
    plt.suptitle('Predicted values after ' + str(iterations) + ' iterations')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    # floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(generatingFunction, 0.05, 0.01, a, b)
    floatAndProjections(X_test[:,0:1], X_test[:,1:2], X_test[:,2:3], col=y_predict_colors, main_alph=0.5, proj_alph=0.01)


    plt.show()

    coef = clf.coef_
    intercept = clf.intercept_
    
    print("Plotting hyperplane")
    coef0 = coef[0,0]
    coef1 = coef[0,1]
    coef2 = coef[0,2]
    
    icpt = intercept[0]
    
    m1 = coef0 / -coef2
    m2 = coef1 / -coef2
    c = icpt / -coef2

    def plane(xi, yi):
        print("Plane")
        # return m1*xi + m2*yi + c
        return 0*xi + 0*yi + 0

    x0 = -1
    x1 = 1
    y0 = -1
    y1 = 1
    
    XQ = np.array([x0, x1])
    YQ = np.array([y0, y1])
    XG, YG = np.meshgrid(XQ, YQ)
    ZG = 0*XG + 0*YG
    # ZG = plane(XG, YG)

          
    #ax = plt.axes(projection='3d')
    plt.suptitle('Hyperplane')
    print("Plotting")
    ax = plt.axes(projection='3d')
    ax.plot_surface(XG, YG, ZG, alpha=1)
    print("Plot show")
    plt.show()

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
    
    plt.show()

'''



