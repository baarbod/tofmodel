# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='tofmodel',
    version='2.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tof = tofmodel.cli:main',
        ],
    }
)