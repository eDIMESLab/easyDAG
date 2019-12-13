#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
  from setuptools import setup
  from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = 'easyDAG'
DESCRIPTION = 'simple tool for deferred pipeline execution and manipulation'
URL = 'https://github.com/eDIMESLab/easyDAG'
EMAIL = ['enrico.giampieri@unibo.it', 'nico.curti2@unibo.it']
AUTHOR = ['Enrico Giampieri', 'Nico Curti']
REQUIRES_PYTHON = '>=3.4'
VERSION =  "0.1.0"
KEYWORDS = 'DAG'

README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'easyDAG', '__version__.py')

LONG_DESCRIPTION = DESCRIPTION

# parse version variables and add them to command line as definitions


setup(
  name                          = NAME,
  version                       = VERSION,
  description                   = DESCRIPTION,
  long_description              = LONG_DESCRIPTION,
  long_description_content_type = 'text/markdown',
  author                        = AUTHOR,
  author_email                  = EMAIL,
  maintainer                    = AUTHOR,
  maintainer_email              = EMAIL,
  python_requires               = REQUIRES_PYTHON,
  url                           = URL,
  download_url                  = URL,
  keywords                      = KEYWORDS,
  packages                      = find_packages(include=['easyDAG', 'easyDAG.*'], exclude=('test', 'testing')),
  #include_package_data          = True, # no absolute paths are allowed
  platforms                     = 'any',
  classifiers                   = [
                                   #'License :: OSI Approved :: GPL License',
                                   'Programming Language :: Python',
                                   'Programming Language :: Python :: 3',
                                   'Programming Language :: Python :: 3.6',
                                   'Programming Language :: Python :: Implementation :: PyPy'
                                  ],
  license                       = 'MIT'
)
