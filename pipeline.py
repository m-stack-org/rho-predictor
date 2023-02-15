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
    normalize = True
    old = False

    refsoapfile = 'data/reference_soap_norm.npz' if normalize else 'data/reference_soap.npz'
    if old:
        refsoapfile = 'data/reference_old.npz'
    refqfile    = 'data/reference_q.npy'
    weightsfile = 'data/weights.npz'
    basis = 'ccpvqz jkfit'

    # Load the molecule
    mol = qstack.compound.xyz_to_mol(molfile, basis=basis)

    # Compute λ-SOAP for the target molecule
    # TODO load the parameters from a file
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
    soap = generate_lambda_soap_wrapper(molfile, rascal_hypers, elements, normalize)
    if old:
        soap = equistore.io.load('ethane.npz')

    # Load regression weights
    weights = equistore.io.load(weightsfile)
    ref_q = np.load(refqfile)

    # Load the averages
    # TODO convert these into a TensorMap
    averages = {q: np.load('data/AVERAGES/'+data.elements.ELEMENTS[q]+'.npy') for q in elements}

    # Load λ-SOAP for the reference environments
    soap_ref = equistore.io.load(refsoapfile)

    # Compute the kernel
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    c_tm = compute_prediction(mol, kernel, weights, averages)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)
    print(c[:16])

    # Save the prediction
    np.savetxt(molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, molfile+'.molden')


if __name__=='__main__':
    main()
