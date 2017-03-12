#!/usr/bin/env python
import sacc
import numpy as np

## demonstrate splitting sacc into three files

s=sacc.SACC.loadFromHDF("test.sacc")

## save separately into three files
s.mean.saveToHDFFile("split.mean.sacc")
s.precision.saveToHDFFile("split.precision.sacc")
# we now save into main file, it will contain pointers for external files for
# mean and precision
s.saveToHDF("split.main.sacc",save_mean=False, mean_filename="split.mean.sacc",
                              save_precision=False, precision_filename="split.precision.sacc")

## we can now reload. It will reconsitute from three files.
sn=sacc.SACC.loadFromHDF("split.main.sacc")
sn.printInfo()

