from setuptools import setup, find_packages

setup(
    name='legrad',
    description='Naive automatic differentiation implementation library',
    author='Lucas Cardozo',
    py_modules=["legrad"],
    python_requires='>=3.6',
    install_requires=[
        'numpy >=1.12',
    ],
    url='https://github.com/lecardozo/legrad',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ]
)