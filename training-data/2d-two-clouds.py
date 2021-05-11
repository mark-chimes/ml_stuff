# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:17:24 2021

@author: mark.chimes
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

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
# Two sets of data: a and b. Horizontal x vertical y

lin_dist = 0.2 # move the two data sets apart
sig = 0.1

# datapoints per group
Na = 1000
Nb = 1000

mu_ax, mu_ay, sigma_ax, sigma_ay  = -lin_dist, -lin_dist, sig, sig*2 # mean and standard deviation
mu_bx, mu_by, sigma_bx, sigma_by  = lin_dist, lin_dist, sig, sig*2 # mean and standard deviation

sax = np.random.normal(mu_ax, sigma_ax, Na)
say = np.random.normal(mu_ay, sigma_ay, Na)
sbx = np.random.normal(mu_bx, sigma_bx, Na)
sby = np.random.normal(mu_by, sigma_by, Na)

#%% Plot Base Data

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
plt.scatter(sax, say, marker='.', color='red')
plt.scatter(sbx, sby, marker='.', color='blue')
plt.show()