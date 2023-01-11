#!/usr/bin/env python3

import numpy as np
import ase.io
import equistore.io
from lsoap import generate_lambda_soap_wrapper





def kernel_nm(mol, soap, soap_ref, ref_q):
    #(M, llmax, lmax, nel,
    #el_dict, ref_elements,
    #kernel_size, kernel_sparse_indices,
    #power, power_ref,
    #atom_counting, atomicindx,
    #imol=None):

    lmax = 6 # TODO

    n = mol.get_global_number_of_atoms()
    kernel = []
    for _ in range(n):
        kernel.append( {l: [] for l in range(lmax+1)})
    print(kernel)


    #k_NM = np.zeros(kernel_size, float)

    for iat, q in enumerate(mol.numbers):
        print(q)

        for l in range(lmax+1):
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, iat))
            vals  = block.values[isamp,:,:]

            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            #irefs = np.where(ref_q==q)[0]
            #for iref in irefs:
            #    isamp_ref = block_ref.samples.position((iref,))
            #    vals_ref  = block_ref.values[isamp_ref,:,:]
            #    k = vals @ vals_ref.T
            #    kernel[iat][l].append(k)
            #kernel[iat][l] = np.array(kernel[iat][l])
            vals_ref  = block_ref.values
            kernel[iat][l] = np.einsum('rmx,Mx->rMm', vals_ref, vals) # maybe M and m should be transposed... TODO

    for i in range(n):
        for l in range(lmax+1):
            print(kernel[i][l].shape)
            print(kernel[i][l])
            print()


#    for q in ref_q:
#        pass
#    for iref in range(M):
#        iel = ref_elements[iref]
#        q   = el_dict[iel]
#        for iatq in range(atom_counting[iel]):
#            iat = atomicindx[iel,iatq]
#            ik0 = kernel_sparse_indices[iref,iatq,0,0,0]
#            for l in range(lmax[q]+1):
#
#                if imol is None:
#                    powert = power[l][iat]
#                else:
#                    powert = power[l][imol,iat]
#                powerr = power_ref[l][iref]
#
#                msize = 2*l+1
#                if l==0:
#                    ik = kernel_sparse_indices[iref,iatq,l,0,0]
#                    k_NM[ik] = np.dot(powert,powerr)**zeta
#                else:
#                    kern = np.dot(powert,powerr.T) * k_NM[ik0]**(float(zeta-1)/zeta)
#                    for im1 in range(msize):
#                        for im2 in range(msize):
#                            ik = kernel_sparse_indices[iref,iatq,l,im1,im2]
#                            k_NM[ik] = kern[im2,im1]
#    return k_NM



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


