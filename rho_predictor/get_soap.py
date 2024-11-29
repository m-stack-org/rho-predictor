#!/usr/bin/env python3

import argparse
import ase.io
import metatensor
from rho_predictor.lsoap import generate_lambda_soap_wrapper, make_rascal_hypers


def main():

    parser = argparse.ArgumentParser(description='Compute λ-SOAP representation for a given structure.')
    parser.add_argument('--mol',           required=True,                type=str,   help='The path to the xyz file with the molecular structure')
    parser.add_argument('--save',          required=True,                type=str,   help='The path to save the output')
    parser.add_argument('--elements',      default=[1,6,7,8], nargs='+', type=int,   help='The elements contained in the database')
    parser.add_argument('--rcut',          default=4.0,                  type=float, help='Cutoff distance (Å)')
    parser.add_argument('--ncut',          default=8,                    type=float, help='Radial cutoff')
    parser.add_argument('--lcut',          default=6,                    type=float, help='Angular cutoff')
    parser.add_argument('--sigma',         default=0.3,                  type=float, help='Gaussian width (Å)')
    parser.add_argument('--basis',         default=None,                 type=str,   help='Basis set to strip the representation')
    parser.add_argument('--dontnormalize', action='store_true',                      help='If skip normalization')
    parser.add_argument('--gradient',      action='store_true',                      help='If compute gradient')
    args = parser.parse_args()
    print(vars(args))

    asemol = ase.io.read(args.mol)
    if args.basis:
        import qstack
        import qstack.equio
        mol = qstack.compound.xyz_to_mol(args.mol, basis=args.basis, ignore=True)
        lmax = {q: max(qstack.equio._get_llist(q, mol)) for q in asemol.get_atomic_numbers()}
    else:
        lmax = None

    rascal_hypers = make_rascal_hypers(args.rcut, args.ncut, args.lcut, args.sigma)
    soap = generate_lambda_soap_wrapper(asemol, rascal_hypers, neighbor_species=args.elements,
                                        normalize=(not args.dontnormalize), lmax=lmax,
                                        gradients=(['positions'] if args.gradient else None))
    metatensor.save(args.save if args.save[-4:]=='.npz' else args.save+'.npz', soap)


if __name__ == '__main__':
    main()
