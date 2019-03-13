#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for modularity-density.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import require, VersionConflict
from setuptools import setup, find_packages

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

with open("pypi_project_description.md", "r") as readme_file:
    readme = readme_file.read()

# specifications to be reflected on PyPi page
reqs = ['numpy>=1.15.1',
'scipy>=1.1.0',
'networkx>=2.2']

if __name__ == "__main__":
    setup(name='modularitydensity',
        version='0.0.6',
        description='Run modularity density-based clustering',
        long_description=readme,
        long_description_content_type='text/markdown',
        license='MIT License (MIT License)',
        url='https://github.com/ckmanalytix/modularity-density',
        packages=find_packages(),
        install_requires=reqs,
        classifiers=['Programming Language :: Python'])
