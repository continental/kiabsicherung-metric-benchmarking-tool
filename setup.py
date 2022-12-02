# Setup

from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.1'

here = path.abspath(path.dirname(__file__))

# get the dependencies
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().split('\n')

install_requires = [x.strip() for x in requirements]

setup(
    name='kia_mbt',
    version=__version__,
    description='KIA Metric Benchmarking Tool.',
    license='internal',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
    ],
    keywords='',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'kia_mbt = kia_mbt.main:main'
        ]
    },
    author='Christian Hellert',
    install_requires=install_requires,
    author_email='christian.hellert@continental.de'
)
