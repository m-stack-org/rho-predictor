refsoapfile = 'data/reference_500.npz'
weightsfile = 'data/weights.npz'
averagefile = 'data/averages.npz'

basis = 'ccpvqz jkfit'
elements = [1, 6, 7, 8]

rascal_hypers = {
    "cutoff": 4.0,
    "max_radial": 8,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
