#!/usr/bin/env python

from setuptools import setup

version = '1.0'

setup(
    name='ptmc',
    version=version,
    description='Python Tif Motion Correction: Multipage tif file manipulation and analysis',
    long_description='See https://github.com/kr-hansen/ptmc',
    author='kr-hansen',
    author_email='kyleh@bu.edu',
    url='https://github.com/kr-hansen/ptmc',
    packages=['ptmc'],
    install_requires=open('requirements.txt').read().split('\n'),
)
