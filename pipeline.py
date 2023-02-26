#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import data
import equistore.io
import qstack
from utils.lsoap import generate_lambda_soap_wrapper
from utils.rhoml import compute_kernel, compute_prediction


def main():

    molfile = sys.argv[1]   # "./H6C2____monA_0932.xyz"
    modelname = sys.argv[2] # "bfdb_HCNO"
    # TODO
    # @Osvaldo:
    # could you please make a drop down list with the models
    # from the models/ dir? for now there's only one

    # Get the model
    model = getattr(__import__("models", fromlist=[modelname]), modelname)

    # Load the molecule
    mol = qstack.compound.xyz_to_mol(molfile, basis=model.basis)

    # Compute λ-SOAP for the target molecule
    soap = generate_lambda_soap_wrapper(molfile, model.rascal_hypers, model.elements, normalize=True)

    # Load regression weights
    weights = equistore.io.load(model.weightsfile)

    # Load the averages
    averages = equistore.io.load(model.averagefile)

    # Load λ-SOAP for the reference environments
    soap_ref = equistore.io.load(model.refsoapfile)

    # Compute the kernel
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    c_tm = compute_prediction(mol, kernel, weights, averages)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)

    # Save the prediction
    equistore.io.save(molfile+'.coeff.npz', c_tm)
    np.savetxt(molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, molfile+'.molden')


if __name__=='__main__':
    main()
