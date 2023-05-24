#!/usr/bin/env python3

'''
Loads old PS files and saves the reference PS in the equistore format
'''

import gc
import glob
import numpy as np
import ase.io
import equistore.core as equistore


def convert_to_tmap(pslist, ref_q):

    elements = sorted(set(ref_q))
    keys = [(l, q) for q in elements for l in range(lmax+1)]

    block_samp_label_vals = {}
    block_prop_label_vals = {}
    block_comp_label_vals = {}
    blocks = {key: [] for key in keys}

    for l in range(lmax+1):
        ps = np.load(pslist[l])
        for q in elements:
            key = (l, q)
            idx = np.where(ref_q==q)
            ps_q = ps[idx]
            if l==0:
                ps_q = np.expand_dims(ps_q, 1)
            blocks[key] = ps_q
            block_samp_label_vals[key] = np.array(idx).T
            block_comp_label_vals[key] = np.arange(-l, l+1).reshape(-1,1)
            block_prop_label_vals[key] = np.arange(ps_q.shape[-1]).reshape(-1,1)

    keys.remove((lmax,1))

    tm_labels = equistore.Labels(('spherical_harmonics_l', 'species_center'), np.array(keys))
    block_comp_labels = {key: equistore.Labels(('spherical_harmonics_m',), block_comp_label_vals[key]) for key in keys}
    block_prop_labels = {key: equistore.Labels(('prop_number',),           block_prop_label_vals[key]) for key in keys}
    block_samp_labels = {key: equistore.Labels(('ref_env',),               block_samp_label_vals[key]) for key in keys}
    blocks = {key: equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in keys}
    tensor = equistore.TensorMap(keys=tm_labels, blocks=[blocks[key] for key in keys])
    return tensor

if __name__=='__main__':
    lmax = 5
    pslist = ['PS_REF/PS'+str(l)+'_1000.npy' for l in range(lmax+1)]
    ref_q = np.load('reference_q.npy')
    tensor = convert_to_tmap(pslist, ref_q)
    equistore.save('reference_old.npz', tensor)
