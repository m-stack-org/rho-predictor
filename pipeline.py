#!/usr/bin/env python3

import numpy as np
import ase.io
import equistore.io
import qstack
from lsoap import generate_lambda_soap_wrapper

def count_elements(mylist, dictionary=False):
    if dictionary is False:
        counter = np.zeros(max(mylist)+1)
    else:
        counter = {}
    for key in set(mylist):
        counter[key] = mylist.count(key)
    return counter

def compute_kernel(atom_charges, soap, soap_ref):
    # TODO return equistor tensor
    lmax   = max(np.array([list(x) for x in soap.keys])[:,0])
    kernel = []
    for _ in atom_charges:
        kernel.append( {l: [] for l in range(lmax+1)})

    for iat, q in enumerate(atom_charges):
        # Compute the kernel
        for l in range(lmax+1):
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            vals_ref  = block_ref.values
            kernel[iat][l] = np.einsum('rmx,Mx->rMm', vals_ref, vals) # maybe M and m should be transposed... TODO
        # Normalize with zeta=2; should be in descendant order because l=0 is used
        for l in range(lmax, -1, -1):
            kernel[iat][l] = kernel[iat][l] * kernel[iat][0]
    return kernel




def compute_prediction(mol):

    # TODO reorder p-orbitals after
    # TODO for some reason the old PS are already normalized (norm of self dot product is 1 for any L)
    # c_{iat, l, m, n} = K_{iat/ref, l, m/m1} * w_{ref, l, n, m1}

    c = np.zeros(mol.nao)
    i = 0
    for iat in range(mol.natm):
        q = mol.atom_charge(iat)
        refs = np.where(ref_q == q)[0]


        llist = [mol.bas_angular(bas_id) for bas_id in mol.atom_shell_ids(iat)]
        if llist!=sorted(llist):
            raise NotImplementedError('Cannot work with a basis with L not in order')

        lmax = max(llist)

        print(llist)


        counter = count_elements(llist)


        print(counter)

        exit(0)

        n_tot = np.zeros(lmax+1, dtype=int)  # counter for radial channels for this l

        for bas_id in mol.atom_shell_ids(iat):
            l  = mol.bas_angular(bas_id)
            nc = mol.bas_nctr(bas_id)
            block = weights.block(spherical_harmonics_l=l, element=q)
            msize = 2*l+1
            for n in range(nc):
                id_prop = block.properties.position((n_tot[l],))

                w = block.values[:,:,id_prop]
                k = kernel[iat][l][:,:,:]
                c[i:i+msize] = np.einsum('rmM,rM->m',  k,  w)
                i += msize
                n_tot[l]+=1
    print(c)
    return c



elements = [1, 6, 7, 8]
molfile = "1.xyz"
molfile = "./H6C2____monA_0932.xyz"
basis = 'ccpvqz jkfit'
#basis = 'sto3g'
mol = qstack.compound.xyz_to_mol(molfile, basis=basis)

if 0:
    mol0 = qstack.compound.xyz_to_mol(molfile, basis='ccpvdz')
    dm   = qstack.fields.dm.get_converged_dm(mol0, xc="pbe")
    auxmol, c0 = qstack.fields.decomposition.decompose(mol0, dm, 'cc-pvqz jkfit')
    print(c0)
    exit(0)


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
soap = generate_lambda_soap_wrapper(molfile, rascal_hypers, elements)

# Load regression weights
weights = equistore.io.load('weights.npz')
ref_q = np.load('reference_q.npy')

compute_prediction(mol)

# Load λ-SOAP for the reference environments
soap_ref = equistore.io.load('reference_soap.npz')



# Compute the kernel
kernel = compute_kernel(mol.atom_charges(), soap, soap_ref)
np.save('kernel.npy', kernel)

# Compute the prediction
compute_prediction(mol)
pass

# Correct the N
pass

# Compute some properties
pass


