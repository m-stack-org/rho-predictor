#!/usr/bin/env python3

import sys
from datetime import datetime
import numpy as np
from pyscf import data
import equistore.io
import qstack
from utils.lsoap import generate_lambda_soap_wrapper
from utils.rhoml import compute_kernel, compute_prediction

def mylog(s, printlvl=0):
    if printlvl > 0:
        print(datetime.now(), s)

def main():

    molfile = sys.argv[1]   # "./H6C2____monA_0932.xyz"
    modelname = sys.argv[2] # "bfdb_HCNO"
    printlvl = 1

    # Get the model
    mylog('# Get the model', printlvl)
    model = getattr(__import__("models", fromlist=[modelname]), modelname)

    # Load the molecule
    mylog('# Load the molecule', printlvl)
    mol = qstack.compound.xyz_to_mol(molfile, basis=model.basis)

    # Compute λ-SOAP for the target molecule
    mylog('# Compute λ-SOAP for the target molecule', printlvl)
    soap = generate_lambda_soap_wrapper(molfile, model.rascal_hypers, model.elements, normalize=True)

    # Load regression weights
    mylog('# Load regression weights', printlvl)
    weights = equistore.io.load(model.weightsfile)

    # Load the averages
    mylog('# Load the averages', printlvl)
    averages = equistore.io.load(model.averagefile)

    # Load λ-SOAP for the reference environments
    mylog('# Load λ-SOAP for the reference environments', printlvl)
    soap_ref = equistore.io.load(model.refsoapfile)

    # Compute the kernel
    mylog('# Compute the kernel', printlvl)
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    mylog('# Compute the prediction', printlvl)
    c_tm = compute_prediction(mol, kernel, weights, averages)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)

    # Save the prediction
    mylog('# Save the prediction', printlvl)
    equistore.io.save(molfile+'.coeff.npz', c_tm)
    np.savetxt(molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, molfile+'.molden')


if __name__=='__main__':
    main()
