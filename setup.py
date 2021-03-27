from setuptools import setup, find_packages


DESCRIPTION = 'Kuramoto model on graphs'

base_packages = [
    "numpy>=1.16.0",
    "scipy",
]

test_packages = [
    "pytest",
    "mypy",
]


setup(
    name='kuramoto',
    version='0.2.1',
    description=DESCRIPTION,
    author='Fabrizio Damicelli',
    author_email='fabridamicelli@gmail.com',
    url="https://github.com/fabridamicelli/kuramoto",
    packages=find_packages(exclude=['notebooks', 'docs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
    install_requires=base_packages,
    include_package_data=True
)
