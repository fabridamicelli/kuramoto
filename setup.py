from setuptools import setup, find_packages


DESCRIPTION = 'Kuramoto model on graphs'
with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


base_packages = [
    "numpy>=1.16.0",
    "scipy",
    "matplotlib",
]

test_packages = [
    "pytest",
    "mypy",
]


setup(
    name='kuramoto',
    version='0.2.4',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
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
