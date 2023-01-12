#!/usr/bin/env python3

'''
Loads old PS files and saves the reference PS in the equistore format

TODO almost the same as get_old_ref_ps.py -> merge
'''



import gc
import glob
import numpy as np
import ase.io
import equistore.io


def convert_to_tmap(pslist, ref_q):

    elements = sorted(set(ref_q))
    keys = [(l, q) for q in elements for l in range(lmax+1)]

    block_samp_label_vals = {}
    block_prop_label_vals = {}
    block_comp_label_vals = {}
    blocks = {key: [] for key in keys}

    for l in range(lmax+1):
        ps = np.load(pslist[l])

        positions = np.arange(len(ref_q))

        for q in elements:
            key = (l, q)
            idx = np.where(ref_q==q)

            # because the elements are sorted in the old PS files
            idx2 = positions[:len(idx[0])]
            positions = positions[len(idx[0]):]
            ps_q = ps[0][idx2]

            if l==0:
                ps_q = np.expand_dims(ps_q, -2)
            blocks[key] = ps_q
            block_samp_label_vals[key] = np.pad(np.array(idx).T, ((0, 0), (1, 0)), 'constant')
            block_comp_label_vals[key] = np.arange(-l, l+1).reshape(-1,1)
            block_prop_label_vals[key] = np.arange(ps_q.shape[-1]).reshape(-1,1)


    tm_labels = equistore.Labels(('spherical_harmonics_l', 'species_center'), np.array(keys))
    block_comp_labels = {key: equistore.Labels(('spherical_harmonics_m',), block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: equistore.Labels(('prop_number',),           block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(('structure', 'center'),    block_samp_label_vals[key]) for key in blocks}
    blocks = {key: equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in keys}
    tensor = equistore.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor

if __name__=='__main__':
    lmax = 5
    pslist = ['PS_2300/PS'+str(l)+'_2300.npy' for l in range(lmax+1)]
    mol = ase.io.read('H6C2____monA_0932.xyz')
    tensor = convert_to_tmap(pslist, mol.numbers)
    equistore.io.save('ethane.npz', tensor)
