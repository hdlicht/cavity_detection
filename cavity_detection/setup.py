from setuptools import setup, find_packages

setup(
    name='cavity_detection',
    version='0.0.0',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
)