#!/usr/bin/env python3

import sys
import equistore.core as equistore
from rho_predictor.lsoap import generate_lambda_soap_wrapper

molfile = sys.argv[1]
rascal_hypers = {
    "cutoff": 4.0,
    "max_radial": 8,  # Exclusive
    "max_angular": 6,  # Inclusive
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
soap = generate_lambda_soap_wrapper([molfile], rascal_hypers, neighbor_species=[1,6,7,8,16], normalize=True)
equistore.save(sys.argv[2], soap)
