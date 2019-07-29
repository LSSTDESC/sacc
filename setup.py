#!/usr/bin/env python
from setuptools import setup

description = "SACC - the Simons Observatory two-point data format library"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(name="sacc", 
      version="0.1.0",
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simonsobs/sacc",
      author="Simons Observatory",
      author_email="david.alonso@physics.ox.ac.uk",
      install_requires=requirements,
      packages=['sacc'],
)
