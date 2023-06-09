from setuptools import setup, find_packages

setup(
    name='rho-predictor',
    version='0.1.0',
    description='Tool to predict the electronic density of molecules using a SA-GPR model',
    url='https://github.com:lcmd-epfl/rho-predictor.git',
    install_requires=[
        'ase',
        'numpy',
        'scipy',
        'pyscf',
        'wigners',
        'equistore-core @ git+https://github.com/lab-cosmo/equistore.git@e5b9dc365369ba2584ea01e9d6a4d648008aaab8#subdirectory=python/equistore-core',
        'rascaline @ git+https://github.com/lab-cosmo/rascaline.git@87a361487b57345def5bad9cc668dc6f54b158f6',
        'qstack @ git+https://github.com/lcmd-epfl/Q-stack.git@f3cac3699dfe7e997f6c2bac9227a6cf5c3d0975'
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['models/*.json']},
)
