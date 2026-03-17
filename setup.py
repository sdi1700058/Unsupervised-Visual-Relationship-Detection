#!/bin/env python3

from setuptools import setup, find_packages

setup(name='latplan',
      version='0.0.1',
      install_requires=[
          'tensorflow-gpu==1.15.2',
          'keras==2.2.5',
          'h5py==2.10.0',
          'numpy>=1.16.0,<1.24.0',
          'scipy==1.4.1',
          'scikit-image',
          'imageio',
          'pillow',
          'matplotlib',
          'progressbar2',
          'keras-adabound==0.6.0',
          'keras-rectified-adam==0.9.0',
          'timeout_decorator',
          'ansicolors',
          'protobuf==3.19.6',],
      packages=find_packages(),
      include_package_data=True,
)
