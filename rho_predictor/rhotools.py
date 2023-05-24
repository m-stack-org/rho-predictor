import os
from types import SimpleNamespace
import json
import qstack
import qstack.equio
from rho_predictor.lsoap import make_rascal_hypers


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


def load_model(modelfile):
    default_dir = os.path.dirname(__file__)+'/models/'

    if os.path.isfile(default_dir+modelfile):
        modelfile = default_dir+modelfile
    elif os.path.isfile(default_dir+modelfile+'.json'):
        modelfile = default_dir+modelfile+'.json'
    elif os.path.isfile(modelfile+'.json'):
        modelfile = modelfile+'.json'
    elif not os.path.isfile(modelfile):
        raise RuntimeError(f'Cannot find the model file {modelfile}')

    with open(modelfile, "r") as f:
        model = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    model.rascal_hypers = make_rascal_hypers(model.cutoff, model.max_radial, model.max_angular, model.atomic_gaussian_width)
    return model
