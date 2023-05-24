import numpy as np
import equistore.core as equistore


def compute_kernel(soap, soap_ref):

    # Compute kernel
    keys = keys_intersection(soap, soap_ref)
    kblocks = []
    tm_labels = equistore.Labels(soap.keys.names, keys)
    for key in tm_labels:
        sblock = soap.block(key)
        rblock = soap_ref.block(key)
        samples = equistore.Labels(soap.sample_names[1:2], sblock.samples.asarray()[:,[1]])
        components = [equistore.Labels([soap.components_names[0][0]+str(i)], sblock.components[0].asarray()) for i in [1,2]]
        values = np.einsum('rmx,aMx->aMmr', rblock.values, sblock.values)
        kblock = equistore.TensorBlock(values=values, samples=samples, components=components, properties=rblock.samples)
        if sblock.has_gradient('positions'):
            sgrad = sblock.gradient('positions')
            gvalues = np.einsum('rmx,adMx->adMmr', rblock.values, sgrad.values)
            kgrad = equistore.TensorBlock(values=gvalues, samples=sgrad.samples,
                                          components=sgrad.components[0:1]+components,
                                          properties=kblock.properties)
            kblock.add_gradient('positions', kgrad)
        kblocks.append(kblock)
    kernel = equistore.TensorMap(keys=tm_labels, blocks=kblocks)

    # Normalize with zeta=2, mind the loop direction
    elements = np.unique(tm_labels.asarray()[:,1])
    lmax = {q: max(map(lambda x: x[0], filter(lambda x: x[1]==q, keys))) for q in elements}
    for q in elements:
        k0block = kernel.block(spherical_harmonics_l=0, species_center=q)
        for l in range(lmax[q], -1, -1):
            k1block = kernel.block(spherical_harmonics_l=l, species_center=q)

            if k1block.has_gradient('positions'):
                k0grad = k0block.gradient('positions')
                k1grad = k1block.gradient('positions')
                # gradient samples are ['sample', 'structure', 'atom']
                # => reshaping gives [sample, atom, direction, spherical_harmonics_m1, spherical_harmonics_m2, ref_env]
                g0 = k0grad.values.reshape(len(k0block.samples), -1, *k0grad.values.shape[1:])
                g1 = k1grad.values.reshape(len(k1block.samples), -1, *k1grad.values.shape[1:])
                k1g0 = np.einsum('aMmr,abdr->abdMmr', k1block.values, g0[:,:,:,0,0,:])
                k0g1 = np.einsum('ar,abdMmr->abdMmr', k0block.values[:,0,0,:], g1)
                k1grad.values[...] = (k1g0+k0g1).reshape(*k1grad.values.shape)
            k1block.values[...] *= k0block.values[...]

    return kernel


def keys_intersection(t1, t2):
    keys1 = {tuple(key) for key in t1.keys}
    keys2 = {tuple(key) for key in t2.keys}
    return np.array(sorted(keys1 & keys2, key=lambda x: x[::-1]))


def compute_prediction(kernel, weights, averages=None):

    cblocks = []
    keys = keys_intersection(kernel, weights)
    tm_labels = equistore.Labels(kernel.keys.names, keys)
    for key in tm_labels:
        kblock = kernel.block(key)
        wblock = weights.block(key)
        values = np.einsum('amMr,rMn->amn', kblock.values, wblock.values)
        if averages and key[0]==0:
            values += averages.block(species_center=key[1]).values
        cblock = equistore.TensorBlock(values=values, samples=kblock.samples,
                                       components=wblock.components, properties=wblock.properties)

        if kblock.has_gradient('positions'):
            kgrad = kblock.gradient('positions')
            gvalues = np.einsum('admMr,rMn->admn', kgrad.values, wblock.values)
            cgrad = equistore.TensorBlock(values=gvalues, samples=kgrad.samples,
                                          components=kgrad.components[0:1]+cblock.components, properties=cblock.properties)
            cblock.add_gradient('positions', cgrad)
        cblocks.append(cblock)

    coeffs = equistore.TensorMap(keys=tm_labels, blocks=cblocks)
    return coeffs
