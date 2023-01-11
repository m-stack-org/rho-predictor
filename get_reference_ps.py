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
import equistore.io


def get_ref_idx(mollist, refs):
    mols = []
    idx_mol = []
    idx_atm = []
    for i,file in enumerate(mollist):
        mol = ase.io.read(file)
        n = mol.get_global_number_of_atoms()
        mols.append(mol)
        idx_mol += [i] * n
        idx_atm += range(n)
    idx = []
    for ref in refs:
        idx.append((idx_mol[ref], idx_atm[ref]))
    return mols, idx


def prepare_blocks_dict(idx):

    elements = set([q for mol in mols for q in mol.numbers])
    keys = [(l, q) for q in elements for l in range(lmax+1)]

    tm_labels = None
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_label_vals = {key: [] for key in keys}
    blocks = {key: [] for key in keys}

    ref_q = np.zeros(len(refs), dtype=int)

    for iref, ref in enumerate(idx):
        print(iref)
        mol_id, atom_id  = ref
        q = mols[mol_id][atom_id].number
        ref_q[iref] = q
        tensor = equistore.io.load(npzlist[mol_id])

        for l in range(lmax+1):
            key = (l, q)
            block = tensor.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, atom_id))
            blocks[key].append(np.copy(block.values[isamp,:,:]))
            block_samp_label_vals[key].append(iref)
            if key not in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties
            if tm_labels is None:
                tm_labels = equistore.Labels(tensor.keys.names, np.array(keys))

        del tensor
        gc.collect()

    for key in keys:
        block_samp_label = equistore.Labels(['ref_env'], np.array(block_samp_label_vals[key]).reshape(-1,1))
        blocks[key] = equistore.TensorBlock(values=np.array(blocks[key]),
                                            samples=block_samp_label,
                                            components=block_comp_labels[key],
                                            properties=block_prop_labels[key])

    tensor = equistore.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor, ref_q

if __name__=='__main__':
    lmax = 6
    mollist = sorted(glob.glob("bfdb/dim*.xyz")) + sorted(glob.glob("bfdb/*mon*.xyz"))
    npzlist = sorted(glob.glob("bfdb/dim*.npz")) + sorted(glob.glob("bfdb/*mon*.npz"))
    refs = np.loadtxt('bfdb/refs_selection_1000.txt', dtype=int)

    mols, idx = get_ref_idx(mollist, refs)
    tensor, ref_q = prepare_blocks_dict(idx)
    print(tensor)

    equistore.io.save('reference.npz', tensor)
    np.save('reference_q.npy', ref_q)
