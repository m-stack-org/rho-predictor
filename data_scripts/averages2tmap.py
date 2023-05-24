#!/usr/bin/env python3

import numpy as np
from pyscf import data
import equistore.core as equistore
from qstack import equio

elements = [1, 6, 7, 8, 16]
averages = {q: np.load('bfdb_with_s/AVERAGES/'+data.elements.ELEMENTS[q]+'.npy') for q in elements}

tm_label_vals = []
tensor_blocks = []
for q in elements:
    tm_label_vals.append((0,q))
    values = averages[q].reshape(1,1,-1)
    prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
    samp_label_vals = np.array([[0]])
    comp_label_vals = np.array([[0]])
    properties = equistore.Labels(equio.vector_label_names.block_prop, prop_label_vals)
    samples    = equistore.Labels(equio.vector_label_names.block_samp, samp_label_vals)
    components = [equistore.Labels(equio.vector_label_names.block_comp, comp_label_vals)]
    tensor_blocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
tm_labels = equistore.Labels(equio.vector_label_names.tm, np.array(tm_label_vals))
tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

equistore.save('averages.npz', tensor)
