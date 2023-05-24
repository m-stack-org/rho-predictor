#!/usr/bin/env python3

import sys
from datetime import datetime
import numpy as np
import ase.io
from pyscf import data
import equistore.io
import qstack
import qstack.equio
from rho_predictor.lsoap import generate_lambda_soap_wrapper
from rho_predictor.rhoml import compute_kernel, compute_prediction
from rho_predictor.rhotools import correct_N_inplace

def log(s, printlvl=0):
    if printlvl > 0:
        print(datetime.now(), s)

def predict_sagpr(molfile, modelname, charge=0, correct_n = True, working_dir="./", model_dir='./', printlvl=0):

    # Get the model
    log('# Get the model', printlvl)
    model = getattr(__import__("rho_predictor.models", fromlist=[modelname]), modelname) # TODO

    # Load the molecule
    log('# Load the molecule', printlvl)
    mol = qstack.compound.xyz_to_mol(working_dir+molfile, basis=model.basis, charge=charge)
    asemol = ase.io.read(working_dir+molfile)

    # Compute λ-SOAP for the target molecule
    log('# Compute λ-SOAP for the target molecule', printlvl)
    soap = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, model.elements, normalize=True)

    # Load regression weights
    log('# Load regression weights', printlvl)
    weights = equistore.io.load(model_dir+model.weightsfile)

    # Load the averages
    log('# Load the averages', printlvl)
    averages = equistore.io.load(model_dir+model.averagefile)

    # Load λ-SOAP for the reference environments
    log('# Load λ-SOAP for the reference environments', printlvl)
    soap_ref = equistore.io.load(model_dir+model.refsoapfile)

    # Compute the kernel
    log('# Compute the kernel', printlvl)
    kernel = compute_kernel(soap, soap_ref)

    # Compute the prediction
    log('# Compute the prediction', printlvl)
    c_tm = compute_prediction(kernel, weights, averages)

    # Correct the number of electrons
    if correct_n:
        correct_N_inplace(mol, c_tm)

    # Save the prediction
    log('# Save the prediction', printlvl)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)
    equistore.io.save(working_dir+molfile+'.coeff.npz', c_tm)
    np.savetxt(working_dir+molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, working_dir+molfile+'.molden')
    return c


if __name__=='__main__':
    predict_sagpr(sys.argv[1], sys.argv[2], printlvl=1)
