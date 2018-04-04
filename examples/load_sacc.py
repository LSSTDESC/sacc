#!/usr/bin/env python
from __future__ import print_function
import sacc
import numpy as np

s=sacc.SACC.loadFromHDF("test.sacc")
s.printInfo()

## print precision matrix

print ("Precision matrix, first 5 colums:")
## This will invert matrix from covariance on the fly
m=s.precision.precisionMatrix()
for i in range(5):
    for j in range(5):
        print ('%5.5f '%m[i,j],end="")
    print()
    

