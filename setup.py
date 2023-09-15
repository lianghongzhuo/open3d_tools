#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 07/11/2018: 10:27 AM
# File Name  : setup.py

from setuptools import setup, find_packages

__version__ = "0.0.2"
setup(
    name="open3d_tools",
    version=__version__,
    description="open3d tools",
    license="MIT License",
    url="https://github.com/lianghongzhuo/open3d_tools",
    author="Hongzhuo Liang",
    author_email="liang@informatik.uni-hamburg.de",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["open3d", "numpy"]
)
