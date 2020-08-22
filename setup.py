from setuptools import setup, find_packages
from ltc.util.version import get_git_version

setup(
    name='ltc',
    version=get_git_version(),
    description='A local turbulence closure model for the oceanic boundary layer',
    author='Achim Randelhoff',
    packages=find_packages(),
    install_requires=[
        'gsw',
        'numpy',
        'param',
    ],
    extras_require=
        {
            'optional': [
                'xarray',
            ]
        }
)
