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
    author_email='emerson.f.harkin@gmail.com',
    url='https://github.com/efharkin/ez-ephys',
    download_url='https://pypi.org/project/ez-ephys/',
    keywords='neuroscience,electrophysiology',
    packages=['ezephys'],
    install_requires=[
        'matplotlib', 'seaborn', 'numpy', 'neo', 'numba'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ]
)
