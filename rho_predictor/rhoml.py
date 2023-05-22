import numpy as np
import equistore
import qstack.equio as equio


def compute_kernel(soap, soap_ref):
    keys1 = set([tuple(key) for key in soap.keys])
    keys2 = set([tuple(key) for key in soap_ref.keys])
    keys  = sorted(keys1 & keys2, key=lambda x: x[::-1])

    # Create an empty kernel tmap
    kblocks = []
    tm_labels = equistore.Labels(soap.keys.names, np.array(keys))
    for key in tm_labels:
        sblock = soap.block(key)
        rblock = soap_ref.block(key)
        samples = equistore.Labels(soap.sample_names[1:2], sblock.samples.asarray()[:,[1]])
        properties = rblock.samples
        components = [equistore.Labels([soap.components_names[0][0]+str(i)], sblock.components[0].asarray()) for i in [1,2]]
        values = np.zeros((len(samples), len(components[0]), len(components[1]), len(properties)))
        kblocks.append(equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties))
        if sblock.has_gradient('positions'):
            gradient = sblock.gradient('positions')
            kblocks[-1].add_gradient(parameter='positions',
                                     data=np.zeros(gradient.data.shape[:2] + values.shape[1:]),
                                     samples=gradient.samples,
                                     components=gradient.components[0:1]+components)
    kernel = equistore.TensorMap(keys=tm_labels, blocks=kblocks)

    # Compute kernel
    for key in tm_labels:
        sblock = soap.block(key)
        rblock = soap_ref.block(key)
        kblock = kernel.block(key)
        kblock.values[:,:,:,:] = np.einsum('rmx,aMx->aMmr', rblock.values, sblock.values)

        if kblock.has_gradient('positions'):
            sgrad = sblock.gradient('positions')
            kgrad = kblock.gradient('positions')
            kgrad.data[:,:,:,:,:] = np.einsum('rmx,adMx->adMmr', rblock.values, sgrad.data[:,:,:,:])

    # Normalize with zeta=2, mind the loop direction
    elements = np.unique(tm_labels.asarray()[:,1])
    lmax = {q: max(map(lambda x: x[0], filter(lambda x: x[1]==q, keys))) for q in elements}
    for q in elements:
        k0block = kernel.block(spherical_harmonics_l=0, species_center=q)
        for l in range(lmax[q], -1, -1):
            k1block  = kernel.block(spherical_harmonics_l=l, species_center=q)
            if k1block.has_gradient('positions'):
                k0grad = k0block.gradient('positions')
                k1grad = k1block.gradient('positions')
                # gradient samples are ['sample', 'structure', 'atom']
                # => reshaping gives [sample, atom, direction, spherical_harmonics_m1, spherical_harmonics_m2, ref_env]
                g0 = k0grad.data.reshape(len(k0block.samples), -1, *k0grad.data.shape[1:])
                g1 = k1grad.data.reshape(len(k1block.samples), -1, *k1grad.data.shape[1:])
                k1g0 = np.einsum('aMmr,abdr->abdMmr', k1block.values, g0[:,:,:,0,0,:])
                k0g1 = np.einsum('ar,abdMmr->abdMmr', k0block.values[:,0,0,:], g1)
                k1grad.data[...] = (k1g0+k0g1).reshape(*k1grad.data.shape)
            k1block.values[...] *= k0block.values[...]

    return kernel


def compute_prediction(mol, kernel, weights, averages=None):
    elements = set(mol.atom_charges())
    coeffs = equio.vector_to_tensormap(mol, np.zeros(mol.nao))
    for q in elements:
        for l in sorted(set(equio._get_llist(q, mol))):
            wblock = weights.block(spherical_harmonics_l=l, element=q)
            cblock = coeffs.block(spherical_harmonics_l=l, element=q)
            kblock = kernel.block(spherical_harmonics_l=l, species_center=q)
            for sample in cblock.samples:
                cpos = cblock.samples.position(sample)
                kpos = kblock.samples.position(sample)
                cblock.values[cpos,:,:] = np.einsum('mMr,rMn->mn', kblock.values[kpos], wblock.values)
            if averages and l==0:
                cblock.values[:,:,:] = cblock.values + averages.block(element=q).values
    return coeffs
