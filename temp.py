# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""


import numpy as np
import math
from PIL import Image
 
def setWeight(h, p1, p2):
    return math.exp(-1/h*math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2))
  
img = Image.open('image.jpg')
arr = np.asarray(img, dtype='uint8')
arrWeight = []

h =  int(input())
print(len(arr))
for i in range(len(arr)-1):
    arrWeight.append([])
    for j in range(len(arr[i]) - 2):
        arrWeight[i].append(setWeight(h, arr[i][j], arr[i][j+1]))
