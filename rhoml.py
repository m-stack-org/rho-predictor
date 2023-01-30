import numpy as np
import qstack
import qstack.equio


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

    kernel = {}
    elements = sorted(set(atom_charges))
    for l in range(lmax+1):
        for q in elements:
            key = (l, q)
            kernel[key] = []


    for iat, q in enumerate(atom_charges):
        for l in range(lmax+1):
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            vals_ref  = block_ref.values
            pre_kernel = np.einsum('rmx,Mx->rMm', vals_ref, vals)
            # Normalize with zeta=2
            if l==0:
                factor = pre_kernel
            kernel[(l, q)].append( pre_kernel * factor )
    return kernel


def compute_prediction(mol, kernel, weights, averages=None):

    # c_{iat, l, m, n} = K_{iat/ref, l, m/m1} * w_{ref, l, n, m1}

    elements = set(mol.atom_charges())
    coeffs = qstack.equio.vector_to_tensormap(mol, np.zeros(mol.nao))

    for q in elements:
        for l in sorted(set(qstack.equio._get_llist(q, mol))):
            wblock = weights.block(spherical_harmonics_l=l, element=q)
            cblock = coeffs.block(spherical_harmonics_l=l, element=q)
            kblock = kernel[(l,q)]
            for sample in cblock.samples:
                cpos = cblock.samples.position(sample)
                kpos = cpos # TODO
                cblock.values[cpos,:,:] = np.einsum('rmM,rMn->mn', kblock[kpos], wblock.values)

            if averages and l==0:
                cblock.values[:,:,:] = cblock.values + averages[q]

    return coeffs





