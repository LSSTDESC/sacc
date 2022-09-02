#!/usr/bin/env python
from setuptools import setup

description = "SACC - the LSST/DESC summary statistic data format library"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(name="sacc", 
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/LSSTDESC/sacc",
      author="LSST DESC",
      author_email="joezuntz@googlemail.com",
      use_scm_version=True,
      setup_requires=["setuptools_scm<7.0", "setuptools_scm_git_archive"],
      install_requires=requirements,
      packages=['sacc'],
)
