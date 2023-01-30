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
