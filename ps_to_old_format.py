#!/usr/bin/env python3

import sys
import numpy as np
import equistore.io
from get_reference_ps import ps_normalize_inplace


if __name__=='__main__':
    lmax = 5
    qs = [1, 6, 7, 8]
    normalize = True

    tensor = equistore.io.load(f'bfdb/{sys.argv[1]}')
    i = int(sys.argv[2])

    for l in range(lmax+1):
        print(l)
        ps = []
        for q in qs:
            try:
                block = tensor.block(spherical_harmonics_l=l, species_center=q)
                vals = block.values
                if normalize:
                    for samp in vals:
                        ps_normalize_inplace(samp)
                ps.append(vals)
            except:
                pass
        ps = np.expand_dims(np.squeeze(np.vstack(ps)), axis=0)
        np.save(f'new_converted_to_old/PS{l}_{i}.npy', ps)
