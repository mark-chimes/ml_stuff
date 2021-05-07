# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:53:24 2021

@author: mark.chimes
"""
import numpy as np

N = 6

col0 = np.ones(N)
col1 = np.linspace(0,N-1,N)
col2 = np.linspace(N,2*N,N)
matrix = np.column_stack((col0, col1, col2))