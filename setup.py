"""Install ezephys."""

import os
import sys
import re
import setuptools

# Read ezephys version number from ezephys/__init__.py
currdir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(currdir, 'ezephys', '__init__.py'), 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)
    f.close()

# Set minimal install requirements based on Python version.
install_requires = ['matplotlib', 'seaborn', 'numpy']
if sys.version_info.major == 2:
    # Reasons for pinning packages:
    # - llvmlite versions >=0.32 do not support Python 2.
    # - numba version needs to match llvmlite
    # - quantities versions >= 0.13 do not support Python 2.
    # - neo >= 0.9 do not support Python 2.
    install_requires.extend(
        ['numba==0.43.1', 'llvmlite<0.32.0', 'quantities==0.12.5', 'neo<0.9.0']
    )
elif sys.version_info.major == 3:
    install_requires.extend(['numba', 'llvmlite', 'neo'])
else:
    raise RuntimeError(
        'Python version {} is not supported.'.format(sys.version_info.major)
    )

# Read long description from restructured text file.
with open(os.path.join(currdir, 'long_description.rst'), 'r') as f:
    long_description = f.read()
    f.close()

setuptools.setup(
    name='ezephys',
    version=version,
    description='Tools for working with electrophysiological data.',
    long_description=long_description,
    license='MIT',
    author='Emerson Harkin',
    author_email='emerson@efharkin.com',
    url='https://github.com/efharkin/ez-ephys',
    download_url='https://pypi.org/project/ez-ephys/',
    keywords='neuroscience,electrophysiology',
    packages=['ezephys'],
    install_requires=install_requires,
    python_requires=(
        '>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*'
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
