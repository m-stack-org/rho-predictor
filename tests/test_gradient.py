#!/usr/bin/env python3

import os
import numpy as np
import ase.io
from rho_predictor.lsoap import generate_lambda_soap_wrapper


def test_gradient():
    path = os.path.dirname(os.path.realpath(__file__))

    xyzfile = path+'/data/H2O.xyz'
    modelname = 'bfdb'
    dr = 1e-4
    max_diff = 1e-6

    model = getattr(__import__("rho_predictor.models", fromlist=[modelname]), modelname) # TODO
    asemol = ase.io.read(xyzfile)

    # get numerical gradient
    r = asemol.get_positions()
    gradnum = [[None, None, None] for _ in r]
    for atom in range(len(r)):
        print(f'{atom+1}/{len(r)}')
        for idir in range(3):
            r0 = r[atom,idir]
            r[atom,idir] = r0 + dr
            asemol.set_positions(r)
            s1 = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, neighbor_species=None, normalize=True)
            r[atom,idir] = r0 - dr
            asemol.set_positions(r)
            s2 = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, neighbor_species=None, normalize=True)
            r[atom,idir] = r0
            asemol.set_positions(r)

            for key, block in s1:
                block.values[...] -= s2[key].values
                block.values[...] *= 0.5/dr
            gradnum[atom][idir] = s1
    print()

    # get analytical gradient
    soap = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, neighbor_species=None, normalize=True, gradients=["positions"])

    # compare
    for key, block in soap:
        print(f'{key=}')
        gblock = soap[key].gradient('positions')
        for gpos, (sample, structure, atom) in enumerate(gblock.samples):
            for idir in range(3):
                gnum  = gradnum[atom][idir][key].values[sample]
                ganal = gblock.data[gpos,idir,:,:]
                diff = np.linalg.norm(gnum-ganal)
                print(f'{sample=} {atom=} {idir=} {diff:e}')
                assert diff < max_diff
        print()


if __name__ == '__main__':
    test_gradient()
