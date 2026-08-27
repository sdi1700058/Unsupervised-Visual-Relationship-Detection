#!/bin/env python3

from setuptools import setup, find_packages

setup(name='latplan',
      version='0.0.1',
      # The cluster pins python/3.6.1 (sh/sherlock_config.sh) because that is
      # what TF 1.15 needs. Until now the floor lived only in a comment, and a
      # py3.7-only line reached Sherlock twice: `from __future__ import
      # annotations` in the planner, then `subprocess.run(capture_output=)`.
      # Declaring it here makes an install fail loudly instead.
      python_requires='>=3.6,<3.8',
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
