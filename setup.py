#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name = "qe2py",
    version = "0.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
)
