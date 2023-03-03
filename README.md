# ρ-predictor

A tool to predict the electronic density of molecules using a SA-GPR model

## Requirements:

- qstack
- equistore
- rascaline

## Installation:

```
pip install git+https://github.com/lcmd-epfl/rho-prediction.git
```

## Usage:
In a script
```
import rho_predictor
rho_predictor.predictor.predict_sagpr('path/to/mol.xyz', 'bfdb_HCNO')
```
or as a cli tool:
```
python -m  rho_predictor.predictor path/to/mol.xyz bfdb_HCNO
```


## TODO
Add details and references to the README
