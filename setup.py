"""Install ezephys."""

import setuptools
import os
import re

currdir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(currdir, 'ezephys', '__init__.py'), 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)

setuptools.setup(
    name='ezephys',
    version=version,
    description='Tools for working with electrophysiological data.',
    license='MIT',
    author='Emerson Harkin',
    author_email='emerson.f.harkin at gmail dot com',
    keywords='neuroscience,electrophysiology',
    packages=['ezephys'],
    install_requires=[
        'matplotlib>=3.1.1', 'numpy>=1.17.2', 'neo>=0.7.2', 'numba>=0.45.1']
)
