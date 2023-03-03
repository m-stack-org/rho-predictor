#!/usr/bin/env python3

import sys
from datetime import datetime
import numpy as np
from pyscf import data
import equistore.io
import qstack
from rho_predictor.lsoap import generate_lambda_soap_wrapper #TODO
from rho_predictor.rhoml import compute_kernel, compute_prediction #TODO

def mylog(s, printlvl=0):
    if printlvl > 0:
        print(datetime.now(), s)

def rho_predictor_sagpr(molfile, modelname, working_dir="./", model_dir='./', printlvl=0):

    # Get the model
    mylog('# Get the model', printlvl)
    model = getattr(__import__("models", fromlist=[modelname]), modelname) # TODO

    # Load the molecule
    mylog('# Load the molecule', printlvl)
    mol = qstack.compound.xyz_to_mol(working_dir+molfile, basis=model.basis)

    # Compute λ-SOAP for the target molecule
    mylog('# Compute λ-SOAP for the target molecule', printlvl)
    soap = generate_lambda_soap_wrapper(working_dir+molfile, model.rascal_hypers, model.elements, normalize=True)

    # Load regression weights
    mylog('# Load regression weights', printlvl)
    weights = equistore.io.load(model_dir+model.weightsfile)

    # Load the averages
    mylog('# Load the averages', printlvl)
    averages = equistore.io.load(model_dir+model.averagefile)

    # Load λ-SOAP for the reference environments
    mylog('# Load λ-SOAP for the reference environments', printlvl)
    soap_ref = equistore.io.load(model_dir+model.refsoapfile)

    # Compute the kernel
    mylog('# Compute the kernel', printlvl)
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    mylog('# Compute the prediction', printlvl)
    c_tm = compute_prediction(mol, kernel, weights, averages)
    c = qstack.equio.tensormap_to_vector(mol, c_tm)

    # Save the prediction
    mylog('# Save the prediction', printlvl)
    equistore.io.save(working_dir+molfile+'.coeff.npz', c_tm)
    np.savetxt(working_dir+molfile+'.coeff.dat', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, working_dir+molfile+'.molden')
    return c


if __name__=='__main__':
    rho_predictor_sagpr(sys.argv[1], sys.argv[2], printlvl=1)
