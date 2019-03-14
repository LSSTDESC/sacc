#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "SACC [Save All Correlations and Covariances]"

setup(name="sacc", 
      version="0.1.0",
      description=description,
      url="https://github.com/LSSTDESC/sacc",
      author="LSST DESC",
      packages=['sacc'])


setup(name="sacc2", 
      version="0.1.0",
      description=description,
      url="https://github.com/LSSTDESC/sacc",
      author="LSST DESC",
      packages=['sacc2'])
