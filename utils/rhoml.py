import numpy as np
import equistore
import qstack.equio as equio


def kernels_toTMap(atom_charges, kernel):
    tm_label_vals = list(kernel.keys())
    tensor_blocks = []
    for (l, q) in tm_label_vals:
        values = np.array(kernel[(l, q)]).transpose(0,2,3,1)
        prop_label_vals = np.arange(values.shape[-1]).reshape(-1,1)
        samp_label_vals = np.where(atom_charges==q)[0].reshape(-1,1)
        comp_label_vals = np.arange(-l, l+1).reshape(-1,1)
        properties = equistore.Labels(equio.vector_label_names.block_prop, prop_label_vals)
        samples    = equistore.Labels(equio.vector_label_names.block_samp, samp_label_vals)
        components = [equistore.Labels([name], comp_label_vals) for name in equio.matrix_label_names.block_comp]
        tensor_blocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
    tm_labels = equistore.Labels(equio.vector_label_names.tm, np.array(tm_label_vals))
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)
    return tensor


def compute_kernel(atom_charges, soap, soap_ref):
    lmax = max(np.array([list(key) for key in soap.keys])[:,0])
    elements = sorted(set(atom_charges))
    kernel = {key: [] for key in [(l, q) for q in elements for l in range(lmax+1)] }
    for iat, q in enumerate(atom_charges):
        for l in range(lmax+1):
            block = soap.block(spherical_harmonics_l=l, species_center=q)
            isamp = block.samples.position((0, iat))
            vals  = block.values[isamp,:,:]
            block_ref = soap_ref.block(spherical_harmonics_l=l, species_center=q)
            vals_ref  = block_ref.values
            pre_kernel = np.einsum('rmx,Mx->rMm', vals_ref, vals)
            # Normalize with zeta=2
            if l==0:
                factor = pre_kernel
            kernel[(l,q)].append(pre_kernel * factor)
    kernel = kernels_toTMap(atom_charges, kernel)
    return kernel


def compute_prediction(mol, kernel, weights, averages=None):
    elements = set(mol.atom_charges())
    coeffs = equio.vector_to_tensormap(mol, np.zeros(mol.nao))
    for q in elements:
        for l in sorted(set(equio._get_llist(q, mol))):
            wblock = weights.block(spherical_harmonics_l=l, element=q)
            cblock = coeffs.block(spherical_harmonics_l=l, element=q)
            kblock = kernel.block(spherical_harmonics_l=l, element=q)
            for sample in cblock.samples:
                cpos = cblock.samples.position(sample)
                kpos = kblock.samples.position(sample)
                cblock.values[cpos,:,:] = np.einsum('mMr,rMn->mn', kblock.values[kpos], wblock.values)
            if averages and l==0:
                cblock.values[:,:,:] = cblock.values + averages[q]
    return coeffs
