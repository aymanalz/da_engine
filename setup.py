#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Ayman Alzraiee",
    author_email='ayman.alzraiee@gmail.com',
    python_requires='>=3.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python tool for data assimilation",
    entry_points={
        'console_scripts': [
            'da_engine=da_engine.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='da_engine',
    name='da_engine',
    packages=find_packages(include=['da_engine', 'da_engine.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aymanalz/da_engine',
    version='0.1.0',
    zip_safe=False,
)
