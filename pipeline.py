#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import data
import equistore.io
import qstack
from utils.lsoap import generate_lambda_soap_wrapper
from utils.rhoml import compute_kernel, compute_prediction



def main():

    molfile = sys.argv[1] # "./H6C2____monA_0932.xyz"

    # TODO load the parameters from a file

    refsoapfile = 'data/reference_500.npz'
    weightsfile = 'data/weights.npz'
    averagefile = 'data/averages.npz'
    basis = 'ccpvqz jkfit'

    rascal_hypers = {
        "cutoff": 4.0,
        "max_radial": 8,
        "max_angular": 6,
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    elements = [1, 6, 7, 8]

    # ...

    # Load the molecule
    mol = qstack.compound.xyz_to_mol(molfile, basis=basis)

    # Compute λ-SOAP for the target molecule
    soap = generate_lambda_soap_wrapper(molfile, rascal_hypers, elements, normalize=True)

    # Load regression weights
    weights = equistore.io.load(weightsfile)

    # Load the averages
    averages = equistore.io.load(averagefile)

    # Load λ-SOAP for the reference environments
    soap_ref = equistore.io.load(refsoapfile)

    # Compute the kernel
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    c_tm = compute_prediction(mol, kernel, weights, averages)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)
    print(c[:16])

    # Save the prediction
    equistore.io.save(molfile+'.coeff.npz', c_tm)
    np.savetxt(molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, molfile+'.molden')


if __name__=='__main__':
    main()
