# coding=utf-8
"""
Setup for scikit-surgeryimage
"""

from setuptools import setup, find_packages
import versioneer

# Get the long description
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='scikit-surgeryimage',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='scikit-surgeryimage implements image processing tools and alogrithms for Image Guided Surgery.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/UCL/scikit-surgeryimage',
    author='Stephen Thompson',
    author_email='s.thompson@ucl.ac.uk',
    license='BSD-3 license',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',


        'License :: OSI Approved :: BSD License',


        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='medical imaging',

    packages=find_packages(
        exclude=[
            'doc',
            'tests',
        ]
    ),

    install_requires=[
        'six>=1.10',
        'numpy>=1.11',
        'opencv-contrib-python>=4.1.1.26',
        'scikit-surgerycore>=0.1.0',
    ],

    entry_points={
        'console_scripts': [
            'sksurgeryimage=sksurgeryimage.__main__:main',
        ],
    },
)
