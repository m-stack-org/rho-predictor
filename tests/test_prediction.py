#!/usr/bin/env python3

import os
import numpy as np
from rho_predictor import predict_sagpr


def test_prediction():
    path = os.path.dirname(os.path.realpath(__file__))+'/'
    xyzfile = path+'data/H2O.xyz'
    modelpath = path+'data/H2O_test'
    c = predict_sagpr(xyzfile, modelpath, printlvl=0, write=False, working_dir='', model_dir=path)
    c0 = np.loadtxt('tests/data/H2O.xyz.coeff.dat')
    assert np.allclose(c, c0)


if __name__ == '__main__':
    test_prediction()
