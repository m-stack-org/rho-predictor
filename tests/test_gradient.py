#!/usr/bin/env python3

import os
import numpy as np
import ase.io
import metatensor
from rho_predictor.lsoap import generate_lambda_soap_wrapper
from rho_predictor.rhoml import compute_kernel, compute_prediction
from rho_predictor.rhotools import load_model


def get_num_gradient(asemol, func, dr=1e-4, verbose=True):
    r = asemol.get_positions()
    gradnum = [[None, None, None] for _ in r]
    for atom in range(len(r)):
        if verbose:
            print(f'{atom+1}/{len(r)}')
        for idir in range(3):
            r0 = r[atom,idir]
            r[atom,idir] = r0 + dr
            asemol.set_positions(r)
            s1 = func(asemol)
            r[atom,idir] = r0 - dr
            asemol.set_positions(r)
            s2 = func(asemol)
            r[atom,idir] = r0
            asemol.set_positions(r)

            for key, block in s1:
                block.values[...] -= s2[key].values
                block.values[...] *= 0.5/dr
            gradnum[atom][idir] = s1
    if verbose:
        print()
    return gradnum


def check_gradients(soap, gradnum, max_diff=1e-6, verbose=True):
    for key, block in soap:
        if verbose:
            print(f'{key=}')
        gblock = soap[key].gradient('positions')
        for gpos, (sample, structure, atom) in enumerate(gblock.samples):
            for idir in range(3):
                gnum  = gradnum[atom][idir][key].values[sample]
                ganal = gblock.values[gpos,idir,:,:]
                diff = np.linalg.norm(gnum-ganal)
                if verbose:
                    print(f'{sample=} {atom=} {idir=} {diff:e}')
                assert diff < max_diff
        if verbose:
            print()


def test_gradient_soap():
    path = os.path.dirname(os.path.realpath(__file__))+'/'
    xyzfile = path+'data/H2O.xyz'
    modelpath = path+'data/H2O_test'
    dr = 1e-4
    max_diff = 1e-6

    model = load_model(modelpath)
    asemol = ase.io.read(xyzfile)
    testfunc = lambda x: generate_lambda_soap_wrapper(x, model.rascal_hypers, neighbor_species=None, normalize=True)

    soap = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, neighbor_species=None, normalize=True, gradients=["positions"])
    gradnum = get_num_gradient(asemol, testfunc, dr=dr)
    check_gradients(soap, gradnum, max_diff=max_diff)


def test_gradient_kernel():
    path = os.path.dirname(os.path.realpath(__file__))+'/'
    xyzfile = path+'data/H2O.xyz'
    modelpath = path+'data/H2O_test'
    dr = 1e-4
    max_diff = 1e-6

    model = load_model(modelpath)
    asemol = ase.io.read(xyzfile)
    soap_ref = metatensor.load(path+model.refsoapfile)
    def testfunc(x):
        soap = generate_lambda_soap_wrapper(x, model.rascal_hypers, neighbor_species=model.elements, normalize=True)
        kernel = compute_kernel(soap, soap_ref)
        return kernel

    soap = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, model.elements, normalize=True, gradients=["positions"])
    kernel = compute_kernel(soap, soap_ref)
    gradnum = get_num_gradient(asemol, testfunc, dr=dr)
    check_gradients(kernel, gradnum, max_diff=max_diff)


def test_gradient_prediction():
    path = os.path.dirname(os.path.realpath(__file__))+'/'
    xyzfile = path+'data/H2O.xyz'
    modelpath = path+'data/H2O_test'
    dr = 1e-4
    max_diff = 1e-6

    model = load_model(modelpath)
    asemol = ase.io.read(xyzfile)
    soap_ref = metatensor.load(path+model.refsoapfile)
    weights = metatensor.load(path+model.weightsfile)
    def testfunc(x):
        soap = generate_lambda_soap_wrapper(x, model.rascal_hypers, neighbor_species=model.elements, normalize=True)
        kernel = compute_kernel(soap, soap_ref)
        c = compute_prediction(kernel, weights)
        return c

    soap = generate_lambda_soap_wrapper(asemol, model.rascal_hypers, model.elements, normalize=True, gradients=["positions"])
    kernel = compute_kernel(soap, soap_ref)
    c = compute_prediction(kernel, weights)
    gradnum = get_num_gradient(asemol, testfunc, dr=dr)
    check_gradients(c, gradnum, max_diff=max_diff)


if __name__ == '__main__':
    test_gradient_soap()
    test_gradient_kernel()
    test_gradient_prediction()
