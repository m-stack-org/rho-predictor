#!/usr/bin/env python3

import numpy as np
from pyscf import data
import equistore.io
import qstack
from lsoap import generate_lambda_soap_wrapper
from get_reference_ps import ps_normalize_inplace
from rhoml import compute_kernel, compute_prediction


def normalize_tensormap(soap):
    for key, block in soap:
        for samp in block.samples:
            isamp = block.samples.position(samp)
            ps_normalize_inplace(block.values[isamp,:,:])


def main():

    #molfile = "1.xyz"
    molfile = "./H6C2____monA_0932.xyz"
    #molfile = "./H6C2.xyz"
    moldenfile = 'H6C2'
    normalize = False
    compare = False
    old = True

    refsoapfile = 'reference_soap_norm.npz' if normalize else 'reference_soap.npz'
    if old:
        refsoapfile = 'reference_old.npz'
    refqfile    = 'reference_q.npy'
    weightsfile = 'weights.npz'
    basis = 'ccpvqz jkfit'

    if compare:
        mol0 = qstack.compound.xyz_to_mol(molfile, basis='ccpvdz')
        dm   = qstack.fields.dm.get_converged_dm(mol0, xc="pbe")
        auxmol, c0 = qstack.fields.decomposition.decompose(mol0, dm, 'cc-pvqz jkfit')
        qstack.fields.density2file.coeffs_to_molden(auxmol, c0, moldenfile+'_ref.molden')
        exit(0)

    # Load the molecule
    mol = qstack.compound.xyz_to_mol(molfile, basis=basis)

    # Compute λ-SOAP for the target molecule
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
    soap = generate_lambda_soap_wrapper(molfile, rascal_hypers, elements)
    if normalize:
        normalize_tensormap(soap)
    if old:
        soap = equistore.io.load('ethane.npz')

    # Load regression weights
    weights = equistore.io.load(weightsfile)
    ref_q = np.load(refqfile)

    # Load the averages
    averages = {q: np.load('AVERAGES/'+data.elements.ELEMENTS[q]+'.npy') for q in elements}

    # Load λ-SOAP for the reference environments
    soap_ref = equistore.io.load(refsoapfile)

    # Compute the kernel
    kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)

    # Compute the prediction
    c = compute_prediction(mol, kernel, weights, averages)
    print(c[:16])

    # Save the prediction
    np.savetxt('cccc', c)
    qstack.fields.density2file.coeffs_to_molden(mol, c, moldenfile+'.molden')


if __name__=='__main__':
    main()
