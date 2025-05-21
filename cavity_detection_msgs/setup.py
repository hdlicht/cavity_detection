from setuptools import setup, find_packages

setup(
    name='cavity_detection_api',
    version='0.0.0',
    packages=find_packages(where="scripts"),
    package_dir={'': 'scripts'},
    install_requires=[],
)
