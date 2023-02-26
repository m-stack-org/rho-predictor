#!/usr/bin/env python3

import numpy as np
from pyscf import gto
import equistore.io
from qstack import equio
import qstack

weights = np.load('../bfdb/weights_NEW_M500_trainfrac1.0_reg1e-06_jit1e-08.npy')
qs = np.load('../data/reference_500_q.npy')

mol = gto.Mole()
mol.atom = [(q, np.random.random(3)) for q in qs]
mol.basis = 'ccpvqz jkfit'
try:
    mol.build()
except:
    mol.charge = -1
    mol.build()

weights = qstack.tools.gpr2pyscf(mol, weights)
tensor = equio.array_to_tensormap(mol, weights)
print(np.linalg.norm(weights - equio.tensormap_to_array(mol, tensor)))
equistore.io.save('weights.npz', tensor)
