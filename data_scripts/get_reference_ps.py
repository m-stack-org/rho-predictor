#!/usr/bin/env python3

'''
Input:
    - list of xyz
    - list of npz containing λ-soap
    - list of reference environments (continuous numbering)
Output:
    - one big npz containing λ-soap
'''

import gc
import glob
import numpy as np
import ase.io
import equistore.core as equistore
from rho_predictor.lsoap import ps_normalize_inplace

def get_ref_idx(mollist, refs):
    mols = []
    idx_mol = []
    idx_atm = []
    for i, file in enumerate(mollist):
        mol = ase.io.read(file+'.xyz')
        n = mol.get_global_number_of_atoms()
        mols.append(mol)
        idx_mol += [i] * n
        idx_atm += range(n)
    idx = []
    for ref in refs:
        idx.append((idx_mol[ref], idx_atm[ref]))
    return mols, idx


def merge_ref_ps(idx, normalize=False):

    elements = sorted(set([q for mol in mols for q in mol.numbers]))
    keys = [(l, q) for q in elements for l in range(lmax+1)]

    tm_labels = None
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_label_vals = {key: [] for key in keys}
    blocks = {key: [] for key in keys}

    ref_q = np.zeros(len(refs), dtype=int)

    tensor_keys_names = None
    for iref, ref in enumerate(idx):
        print(iref)
        mol_id, atom_id  = ref
        q = mols[mol_id][atom_id].number
        ref_q[iref] = q
        tensor = equistore.load(mollist[mol_id]+'.npz')

        for l in range(lmax+1):
            key = (l, q)
            block = tensor.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, atom_id))
            vals  = np.copy(block.values[isamp,:,:])
            if normalize:
                ps_normalize_inplace(vals)
            blocks[key].append(vals)
            block_samp_label_vals[key].append(iref)
            if key not in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties
        if not tensor_keys_names:
            tensor_keys_names = tensor.keys.names

        del tensor
        gc.collect()

    keys.remove((lmax,1))
    tm_labels = equistore.Labels(tensor_keys_names, np.array(keys))

    for key in keys:
        block_samp_label = equistore.Labels(['ref_env'], np.array(block_samp_label_vals[key]).reshape(-1,1))
        blocks[key] = equistore.TensorBlock(values=np.array(blocks[key]),
                                            samples=block_samp_label,
                                            components=block_comp_labels[key],
                                            properties=block_prop_labels[key])

    tensor = equistore.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor, ref_q

if __name__=='__main__':
    lmax = 5

    mydir = "bfdb_with_s/"
    mollist = [mydir+f for f in np.loadtxt(mydir+"mollist.txt", dtype=str)]
    refs = np.loadtxt(mydir+'refs_selection_500.txt', dtype=int)

    mols, idx = get_ref_idx(mollist, refs)
    tensor, ref_q = merge_ref_ps(idx, normalize=True)
    print(tensor)

    equistore.save('reference_500.npz', tensor)
    np.save('reference_500_q.npy', ref_q)
