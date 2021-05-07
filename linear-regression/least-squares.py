# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:01:19 2021

@author: mark.chimes
"""

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
xsq = x**2# square the x-values


plt.plot(x, xsq, 'o', label='exact data')

xx = np.linspace(0, 9, 101) # from 0 to 9 with 101 points
yy = xx**2 # use array calculations to get y-values
plt.plot(xx, yy, label='$y = x^2$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()



y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

M = xsq[:, np.newaxis]**[0, 1] 
# Places the values into a "design matrix" M with a column of 1s.
# This column of 1s allows for the calculation of the constant value
# in the least-squares. I.e., this is the y-intercept - if it weren't
# there, the line would have to go through the origin

p, res, rnk, s = lstsq(M, y)
c = p[0]
m = p[1]


plt.plot(x, y, 'o', label='data')

xx = np.linspace(0, 9, 101) # from 0 to 9 with 101 points
yy = c + m*xx**2 # use array calculations to get y-values

plt.plot(xx, yy, label='least squares fit, $y = c + mx^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()


### CODE BELOW JUST FOR EXAMPLE - NOT CORRECT USAGE ###

M = xsq[:, np.newaxis]##**[0, 1] comment out the part that adds the 1s
# If we use a column of zeroes instead of ones, it calculates a single 
# variable, but 

p, res, rnk, s = lstsq(M, y)
m = p[0]

plt.plot(x, y, 'o', label='data')

xx = np.linspace(0, 9, 101)
yy = m*xx**2 # no +c

plt.plot(xx, yy, label='least squares fit, $y = mx^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()
# The fit goes through the origin, but is really poor.