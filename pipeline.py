#!/usr/bin/env python3

import numpy as np
import pyscf
import equistore.io
import qstack
import qstack.equio
from lsoap import generate_lambda_soap_wrapper
from get_reference_ps import ps_normalize_inplace

def normalize_tensormap(soap):
    for key, block in soap:
        for samp in block.samples:
            isamp = block.samples.position(samp)
            ps_normalize_inplace(block.values[isamp,:,:])

def count_elements(mylist, dictionary=False):
    if dictionary is False:
        counter = np.zeros(max(mylist)+1)
    else:
        counter = {}
    for key in set(mylist):
        counter[key] = mylist.count(key)
    return counter

def compute_kernel(atom_charges, soap, soap_ref):
    # TODO return an equistore tensor

    lmax   = max(np.array([list(key) for key in soap.keys])[:,0])
    kernel = [[None for l in range(lmax+1)] for _ in atom_charges]

    for iat, q in enumerate(atom_charges):
        # Compute the kernel
        for l in range(lmax+1):
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            vals_ref  = block_ref.values
            kernel[iat][l] = np.einsum('rmx,Mx->rMm', vals_ref, vals)
        # Normalize with zeta=2; should be in descendant order because l=0 is used
        for l in range(lmax, -1, -1):
            kernel[iat][l] = kernel[iat][l] * kernel[iat][0]
    return kernel


def compute_prediction(mol, kernel, weights, averages=None):

    # c_{iat, l, m, n} = K_{iat/ref, l, m/m1} * w_{ref, l, n, m1}

    n_for_l = {}
    for q in set(mol.atom_charges()):
        llist = qstack.equio._get_llist(q, mol)
        if llist!=sorted(llist):
            raise NotImplementedError('Cannot work with a basis with L not in order')
        n_for_l[q] = count_elements(llist, dictionary=True)

    c = np.zeros(mol.nao)
    i = 0
    for iat in range(mol.natm):
        q = mol.atom_charge(iat)
        for l in n_for_l[q].keys():
            di = (2*l+1) * n_for_l[q][l]
            wblock = weights.block(spherical_harmonics_l=l, element=q)
            c[i:i+di] = np.einsum('rmM,rMn->nm', kernel[iat][l], wblock.values).flatten()
            if l==0 and averages:
                c[i:i+di] += averages[q]
            i += di
    c = qstack.tools.gpr2pyscf(mol, c)
    return c


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
    averages = {q: np.load('AVERAGES/'+pyscf.data.elements.ELEMENTS[q]+'.npy') for q in elements}

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
