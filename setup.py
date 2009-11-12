# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description='''Image Morphology Toolbox

The image morphology toolbox implements the basic binary and
grayscale morphology operations, working with numpy arrays to
hold image data.

This is a pure Python package which is the companion package
to the book "Hands-on Morphological Image Processing."
'''
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Software Development :: Libraries :: Python Modules',
    ]

setup(name='pymorph',
      version='0.91.4',
      description='Image Morphology Toolbox',
      long_description=long_description,
      author='Luis Pedro Coelho',
      author_email='lpc@mcu.edu',
      url='http://luispedro.org/software/pymorph/',
      license='BSD',
      packages=find_packages(),
      )


