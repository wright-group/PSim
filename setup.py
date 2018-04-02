#! /usr/bin/env python3

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

extra_files = []
extra_files.append(os.path.join(here, 'LICENSE'))

with open(os.path.join(here, 'requirements.txt')) as f:
    required = f.read().splitlines()

with open(os.path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='PSim',
    packages=find_packages(),
    package_data={'': extra_files},
    install_requires=required,
    version=version,
    description='The Python numerical simulator for semiconductor excited states.',
    license='MIT',
)
