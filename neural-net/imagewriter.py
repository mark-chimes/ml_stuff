# -*- coding: utf-8 -*-

import cv2
import glob
 
img_array = []
for filename in glob.glob('C:/ML/Projects/neural-net/progfigs/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('visualization.gif',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()