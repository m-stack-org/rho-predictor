from setuptools import setup, find_packages

setup(
    name='rho-predictor',
    version='0.1.0',
    description='Tool to predict the electronic density of molecules using a SA-GPR model',
    url='https://github.com:lcmd-epfl/rho-predictor.git',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['rho_predictor/models/*.json']},
)
