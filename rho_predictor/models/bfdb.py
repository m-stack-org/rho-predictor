refsoapfile = 'data/reference_500_with_sulphur.npz'
weightsfile = 'data/weights_with_sulphur.npz'
averagefile = 'data/averages_with_sulphur.npz'

basis = 'ccpvqz jkfit'
elements = [1, 6, 7, 8, 16]

rascal_hypers = {
    "cutoff": 4.0,
    "max_radial": 8,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
