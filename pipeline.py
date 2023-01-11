#!/usr/bin/env python3

import numpy as np
import ase.io
import equistore.io
from lsoap import generate_lambda_soap_wrapper





def compute_kernel(mol, soap, soap_ref):

    lmax   = max(np.array([list(x) for x in soap.keys])[:,0])
    n      = mol.get_global_number_of_atoms()
    kernel = []
    for _ in range(n):
        kernel.append( {l: [] for l in range(lmax+1)})

    for iat, q in enumerate(mol.numbers):

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




molfile = "1.xyz"
mol = ase.io.read(molfile)
elements = [1, 6, 7, 8]


# Compute λ-SOAP for the target molecule
rascal_hypers = {
    "cutoff": 4.0,
    "max_radial": 8,     # Exclusive
    "max_angular": 6,    # Inclusive
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
soap = generate_lambda_soap_wrapper(molfile, rascal_hypers, elements)



# Load λ-SOAP for the reference environments
soap_ref = equistore.io.load('soap_ref_bfdb.npz')
ref_q = np.load('reference_q.npy')

kernel_nm(mol, soap, soap_ref, ref_q)

# Compute the kernel
l=1
q=6
iat=0 # within the whole mol
m=0
block = soap.block(spherical_harmonics_l=l, species_center=q)
isamp = block.samples.position((0,iat))
icomp = block.components[0].position((m,))
rep = block.values[isamp,icomp] # one l-soap line
pass





# Load regression weights
pass

# Compute the prediction
pass

# Correct the N
pass

# Compute some properties
pass


