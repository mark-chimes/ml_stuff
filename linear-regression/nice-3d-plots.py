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
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm

seed = 444
random.seed(seed)

xlim = [-1, 1]
ylim = [-1, 1]
axlim = [-1, 1]
aylim = [-1, 1]
azlim = [-2, 1]

#%% Generate Data
def generatingFunction(x, y): 
    return x*y - x**2 +x - 2*y**2 + y

N = 1000
mu, sigma = 0, 0.1 # mean and standard deviation
sx = np.random.normal(mu, sigma, N)
sy = np.random.normal(mu, sigma*2, N)
sz_diff = np.random.normal(mu, sigma, N)
target_sz = generatingFunction(sx, sy)
sz = target_sz + sz_diff

#%% Wireframe setup
gx = gy = np.arange(-0.75, 0.75, 0.1)
gX, gY = np.meshgrid(gx, gy)
gZ0 = np.zeros(gX.shape)
gz = np.array(np.ravel(generatingFunction(gX, gY)))
gZ = gz.reshape(gX.shape)

#%% Linear Regression
num_train = 3

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

#%% Regression Evaluation

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_test_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_test_pred))


#%%  Least Squares

col1 = np.ones(N)
desmat = np.column_stack((col1, sx, sy))
sol, res, rnk, sing = lstsq(desmat, sz)

# Wireframe
gz_sol_a =  np.array(sol[1]*np.ravel(gX) + sol[2]*np.ravel(gY))
gZ_sol = sol[0]*np.ones(gX.shape) + gz_sol_a.reshape(gX.shape)


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

def floatAndProjections(x,y,z, main_alph=0.8, proj_alph=0.08, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z,  marker='.', c = -np.sqrt(x**2 + y**2), alpha=main_alph, cmap='hot')
    ax.scatter3D(x, y, 2*z_reflect(v_angle), c = -np.sqrt(x**2 + y**2), alpha=proj_alph, cmap='hot')
    ax.scatter3D(x, y_reflect(h_angle), z, c = -np.sqrt(x**2 + y**2), alpha=proj_alph, cmap='hot')
    ax.scatter3D(x_reflect(h_angle), y, z, c = -np.sqrt(x**2 + y**2), alpha=proj_alph, cmap='hot')
    
def floatAndProjectionsColorMarker(x,y,z, mark='2', col='blue', main_alph=0.8, proj_alph=0.08, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z, alpha=main_alph, marker=mark, color='blue')
    ax.scatter3D(x, y, 2*z_reflect(v_angle), alpha=proj_alph, marker=mark, color='blue')
    ax.scatter3D(x, y_reflect(h_angle), z, alpha=proj_alph, marker=mark,  color='blue')
    ax.scatter3D(x_reflect(h_angle), y, z,alpha=proj_alph, marker=mark, color='blue')


def predictionFloatAndProjections(x,y,z, main_alph=0.5, proj_alph=0.3, v_angle=0, h_angle=0): 
    ax.scatter3D(x, y, z, alpha=0.5, marker='.', color='green')
    ax.scatter3D(x, y, 2*z_reflect(v_angle),  c = -np.sqrt(x**2 + y**2), alpha=0.3, cmap='Greens')
    ax.scatter3D(x, y_reflect(h_angle), z,  c = -np.sqrt(x**2 + y**2), alpha=0.3, cmap='Greens')
    ax.scatter3D(x_reflect(h_angle), y, z,  c = -np.sqrt(x**2 + y**2), alpha=0.3, cmap='Greens')

def predictionWireframeFloatAndPredictions(z, main_alph=0.2, proj_alph=0.05, v_angle=0, h_angle=0):
    Z = -np.sqrt(gX**2 + gY**2)
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    
    def plot_surfy(ix, iy, iz, ialph): 
         ax.plot_surface(ix, iy, iz, facecolors=colors, rcount=rcount, ccount=ccount, alpha=ialph)
         
    plot_surfy(gX, gY, z, 0.2)
    plot_surfy(gX, gY, gZ0+2*z_reflect(v_angle), 0.05)
    plot_surfy(gX, y_reflect(h_angle), z, 0.05)
    plot_surfy(x_reflect(h_angle), gY, z, 0.05)

#%% Flat Plots
plt.suptitle('Input data')
standardFlatLimitsAndLabels(plt)
plt.scatter(sx, sy, marker='.', color='black')
plt.show()

#%% Plot with no extra data

for a, b in angles: 
    # Plot the points.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset (with shadows / projections)')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a,b)
    floatAndProjections(sx, sy, sz, v_angle=a, h_angle=b)
    plt.show()

#%% Training Data Plots

plt.suptitle('Inputs with Chosen Training Points')
standardFlatLimitsAndLabels(plt)
plt.scatter(sx, sy,  marker='o',  color='black')
plt.scatter(X_train[:,0], X_train[:,1], marker='1', color='lightblue')
plt.show()

# Plot the projection 

# Training Data Projection 1
plt.suptitle('Projection of Outputs vs X of Full Dataset and Training Points')
plt.scatter(sx, sz,  marker='o', color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('X')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_train[:,0], y_train,  marker='1', color='blue')
plt.show()

# Training Data Projection 2
plt.suptitle('Projection of Outputs vs Y of Full Dataset and Training Points')
plt.scatter(sy, sz,  marker='o',  color='orange')
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('Y')
h = plt.ylabel('Z')
h.set_rotation(0)
plt.scatter(X_train[:,1], y_train,  marker='1',  color='blue')
plt.show()

#%% Training 3D
for a, b in angles: 
    # Plot the points.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset and Training Points')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.02, 0.01, a, b)
    floatAndProjectionsColorMarker(X_train[:,0], X_train[:,1], y_train, \
                                   '2', 'blue',  1, 0.5, a, b)
    plt.show()

#%% Generating

for a, b in angles: 
    # Plot the points with true generating function.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset and Generating Function\n' + r'without Perturbation')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a,b)
    floatAndProjections(sx, sy, sz, 0.1, 0.03, a, b)
    predictionFloatAndProjections(sx, sy, target_sz, 0.3, 0.03, a, b)
    plt.show()


#%% Generating Surface

for a, b in angles: 
    # Plot the points with true surface.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset and \n' + r'Generating Function Wireframe')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(gZ, 0.2, 0.1, a, b)
    plt.show()


#%% Least Squares Surface

for a, b in angles: 
    # Plot the points with approximated surface.
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset with Least Squares\n' + r'Approximated Surface')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(gZ_sol, v_angle=a, h_angle=b)
    plt.show()

#%% Linear Regression Points

for a, b in angles: 
    # Plot the points with linear regression values
    ax = plt.axes(projection='3d')
    plt.suptitle('Dataset with Linear Regression\n' + r'Approximated Points')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionFloatAndProjections(sx, sy, y_all_pred, v_angle=a, h_angle=b)
    plt.show()


#%%  Linear Regression Surface

for a, b in angles: 
    # Plot the points with approximated surface.
    plt.suptitle('Dataset with Linear Regression\n' + r'Approximated Surface')
    ax = plt.axes(projection='3d')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.3, 0.03, a, b)
    predictionWireframeFloatAndPredictions(yy, v_angle=a, h_angle=b)
    plt.show()

#%%  Linear Regression Surface on Training Data

tx = X_train[:,0]
ty = X_train[:,1]
tz = y_train[:]

for a, b in angles: 
    # Plot the points with approximated surface.
    plt.suptitle('Training Set with Linear Regression\n' + r'Approximated Surface')
    ax = plt.axes(projection='3d')
    standardAxLimitsAndLabels(ax)
    ax.view_init(a, b)
    floatAndProjections(sx, sy, sz, 0.8, 0.03, a, b)
    predictionWireframeFloatAndPredictions(yy, v_angle=a, h_angle=b)
    plt.show()












