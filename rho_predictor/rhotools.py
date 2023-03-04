import qstack
import qstack.equio

def correct_N_inplace(mol, c, N=None):
    '''Correct the number of electron using Lagrange multipliers and unit metric.'''
    if N is None:
        N = mol.nelectron
    q = qstack.equio.vector_to_tensormap(mol, qstack.fields.decomposition.number_of_electrons_deco_vec(mol))
    N0 = sum((c[key].values*q[key].values).sum() for key in q.keys)  # q @ c
    qq = sum((q[key].values**2).sum() for key in q.keys)  # q @ q
    la = (N-N0) / qq
    for key in c.keys:
        c[key].values[:,:,:] += la * q[key].values[:,:,:]
    return c
