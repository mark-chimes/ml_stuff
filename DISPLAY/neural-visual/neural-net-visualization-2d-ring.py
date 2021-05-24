# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:58:18 2021

@author: mark.chimes
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap

#%% Random Seed
# import random
# seed = 444
# random.seed(seed)

#%% Display Limits
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5

x_min_zoom, x_max_zoom = -0.5, 0.5 
y_min_zoom, y_max_zoom = -0.5, 0.5


#%% Generate Data
# Two sets of data: a and b. Horizontal x vertical y

# Ring
Na = 2000
siga = 0.3
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
Nb = 2000
sigb = 0.3
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
test_N = 200
train_N = total - test_N

X = np.concatenate((sax, sbx))
Y = np.concatenate((say, sby))
Z_categorize = np.concatenate((np.zeros(Na), np.ones(Nb))) # class
all_points = np.column_stack((X, Y, Z_categorize))
np.random.shuffle(all_points)
XY = all_points[:,:-1] # first two columns
Z = np.ravel(all_points[:,-1:]) # last column
Z_colors = np.array(['red' if c > 0.5 else 'blue' for c in Z])

XY_train  = XY[:train_N,:]
XY_test  = XY[train_N:,:]

Z_train = Z[:train_N]
Z_test = Z[train_N:]

Z_train_colors = np.array(['red' if x > 0.5 else 'blue' for x in Z_train]).T
Z_test_colors = np.array(['pink' if x > 0.5 else 'cyan' for x in Z_test]).T
Z_test_colors2 = np.array(['red' if x > 0.5 else 'blue' for x in Z_test]).T

plt.suptitle('Training and Test Datasets')
standardFlatLimitsAndLabels(plt)
plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='2', c=Z_train_colors)
plt.scatter(XY_test[:,:1], XY_test[:,1:], marker=',', c=Z_test_colors)

plt.show()

#%% SGD RELU
max_iter = 10000
solver = 'sgd'
activation = 'relu'
layers = 3
neurons = 8
learning_rate = 0.02
random_state = 1

curve_ylim_min = 0
curve_ylim_max = 1

hidden_layers = tuple(neurons for i in range(layers))
mlp = MLPClassifier(solver=solver, activation=activation, \
                    max_iter=max_iter, alpha=1e-5, hidden_layer_sizes=hidden_layers, random_state=random_state, learning_rate_init = learning_rate)
print('Training neural net ' + str(random_state) + ' with ' + mlp.solver + ' / ' + mlp.activation +  ' at lr ' + str(learning_rate) +\
      ' and layers: ' + str(hidden_layers))


mlp.fit(XY_train, Z_train)

Z_predict = mlp.predict(XY_test)

cmap_light = ListedColormap(['cyan', 'pink'])

h = .02  # step size in the mesh
xx, xy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z_mesh = mlp.predict(np.c_[xx.ravel(), xy.ravel()])
Z_mesh_reshape = Z_mesh.reshape(xx.shape)
#plt.figure(figsize=(8, 6))

title = 'Neural Net ' + str(random_state) + ' with ' + mlp.solver + ' / ' + mlp.activation +  ' at lr ' + str(learning_rate) +\
    ' and layers ' + str(hidden_layers)

standardFlatLimitsAndLabels(plt)
plt.suptitle(title)
plt.contourf(xx, xy, Z_mesh_reshape, cmap=cmap_light)
plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='.',  c=Z_train_colors)
plt.show()

standardFlatLimitsAndLabels(plt)
plt.suptitle(title)
plt.contourf(xx, xy, Z_mesh_reshape, cmap=cmap_light)
plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='.', alpha=0.2, c=Z_train_colors)
plt.show()

standardFlatLimitsAndLabels(plt)
plt.suptitle(title)
plt.contourf(xx, xy, Z_mesh_reshape, cmap=cmap_light)
plt.scatter(XY_test[:,:1], XY_test[:,1:], marker='.', alpha=0.8, c=Z_test_colors2)
plt.show()


if solver == 'sgd': 
    plt.ylim([curve_ylim_min, curve_ylim_max])
    plt.suptitle(title)
    plt.plot(mlp.loss_curve_)
    plt.show()

all_coefs = mlp.coefs_
all_intercepts = mlp.intercepts_

#%% Plots

base_alpha = 0.2
h = .02  # step size in the mesh
xx, xy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def relu(x): 
    return np.maximum(0, x)

def neuron_func(inputs, cfs, icpt): 
    my_sum = np.zeros(inputs[0].shape)
    for i, inp in enumerate(inputs): 
        my_sum = my_sum + cfs[i]*inp
    my_sum = my_sum + icpt*np.ones(inputs[0].shape)
    return relu(my_sum)

def plot_layer_outputs_for_neuron(layer, neuron, all_layer_outputs_, all_coefs_): 
    cur_coefs = (all_coefs_[layer])[:, neuron]
    input_count = cur_coefs.size
    scattering = 5
    totmax = np.max(np.abs(cur_coefs))
    
    vchanges = np.floor(scattering*np.abs(cur_coefs)/totmax).astype(int)
    layer_outputs_prev = all_layer_outputs_[layer]
    for j in range(input_count):
        print('j: ' + str(j))
        
        ax = plt.subplot(scattering+1,input_count+1, (scattering-vchanges[j])*(input_count+1)+j+1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if cur_coefs[j] >= 0: 
            plt.contourf(xx, xy, layer_outputs_prev[j], cmap='seismic')
        else: 
            plt.contourf(xx, xy, -layer_outputs_prev[j], cmap='seismic')
    
    title = 'Layer ' + str(layer-1) + ' weights for neuron ' + str(neuron) + ' of layer ' + str(layer)
    plt.suptitle(title)
    plt.show() 
      
all_layer_outputs = []
cur_layer_outputs = [xx, xy]
all_layer_outputs.append(cur_layer_outputs)  

for layer, layer_coefs in enumerate(all_coefs): 
    layer_intercepts = all_intercepts[layer]
    neuron_count = layer_intercepts.size
    layer_outputs_prev = cur_layer_outputs
    cur_layer_outputs = []
    print("Layer " + str(layer) + " Coefs, Intercepts: " + str(layer_coefs.shape) + " : " + str(layer_intercepts.shape) )

    side_size = math.ceil(math.sqrt(layer_intercepts.size))    
    
    for i in range(neuron_count):
        print('i: ' + str(i))
        cur_coefs = layer_coefs[:, i]
        intercept = layer_intercepts[i]
        print("Coef, intercept " + str(i) + ": " + str(cur_coefs) + " : " + str(intercept) )
        Z_mesh = neuron_func(layer_outputs_prev, cur_coefs, intercept)
        
        cur_layer_outputs.append(Z_mesh)
        
        my_alpha = base_alpha/layer_intercepts.size
        ax = plt.subplot(side_size,side_size,i+1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.contourf(xx, xy, Z_mesh, cmap='seismic')
        plt.scatter(XY_train[:,:1], XY_train[:,1:], marker='.', alpha=my_alpha, c=Z_train_colors)
        
    all_layer_outputs.append(cur_layer_outputs)

    title = 'Layer ' + str(layer)
    plt.suptitle(title)
    plt.show()
    
neuron = 0
for layer in range(len(all_coefs)-1,0,-1):        
    print("Layer " + str(layer))
    plot_layer_outputs_for_neuron(layer, neuron, all_layer_outputs, all_coefs)
    cur_coefs = (all_coefs[layer][:,neuron])
    neuron = np.argmax(np.abs(cur_coefs))

