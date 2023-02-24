#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import data
import equistore.io
import qstack

def main():
    basis = 'ccpvqz jkfit'
    molfile = sys.argv[1]
    cfile = sys.argv[2]
    p0 = float(sys.argv[3])
    npoints = int(sys.argv[4])

    mol = qstack.compound.xyz_to_mol(molfile, basis=basis)
    c = np.loadtxt(cfile) # maybe load a tensormap TODO
    # Correct the number of electrons
    c = qstack.fields.decomposition.correct_N(mol, c)
    # Reorder AO
    c = qstack.tools.pyscf2gpr(mol, c)
    np.savetxt(cfile+'.qinput', c)


    with open(molfile+'.in', 'w') as f:

      print(f'''
#control
 theory = isodensity
#

#density
 coef={cfile+'.qinput'}
#

#esp
 grid={molfile}.grid.dat
#

#isodensity
 fancy = 0
 np = {npoints}
 p0 = {p0}
#''', file=f)
      print('$molecule', file=f)
      print('cart', file=f)
      for i in range(mol.natm):
          print(mol.atom_charge(i), *mol.atom_coord(i, unit='angstrom'), file=f)
      print('$end', file=f)




main()
