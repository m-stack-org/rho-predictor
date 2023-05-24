import numpy as np
import equistore


def compute_kernel(soap, soap_ref):

    # Compute kernel
    keys = keys_intersection(soap, soap_ref)
    kblocks = []
    tm_labels = equistore.Labels(soap.keys.names, keys)
    for key in tm_labels:
        sblock = soap.block(key)
        rblock = soap_ref.block(key)
        samples = equistore.Labels(soap.sample_names[1:2], sblock.samples.asarray()[:,[1]])
        properties = rblock.samples
        components = [equistore.Labels([soap.components_names[0][0]+str(i)], sblock.components[0].asarray()) for i in [1,2]]
        values = np.zeros((len(samples), len(components[0]), len(components[1]), len(properties)))
        kblock = equistore.TensorBlock(values=values, samples=samples, components=components, properties=properties)
        kblock.values[:,:,:,:] = np.einsum('rmx,aMx->aMmr', rblock.values, sblock.values)
        if sblock.has_gradient('positions'):
            sgrad = sblock.gradient('positions')
            data = np.einsum('rmx,adMx->adMmr', rblock.values, sgrad.data)
            kblock.add_gradient(parameter='positions', data=data,
                                samples=sgrad.samples, components=sgrad.components[0:1]+components)
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
                g0 = k0grad.data.reshape(len(k0block.samples), -1, *k0grad.data.shape[1:])
                g1 = k1grad.data.reshape(len(k1block.samples), -1, *k1grad.data.shape[1:])
                k1g0 = np.einsum('aMmr,abdr->abdMmr', k1block.values, g0[:,:,:,0,0,:])
                k0g1 = np.einsum('ar,abdMmr->abdMmr', k0block.values[:,0,0,:], g1)
                k1grad.data[...] = (k1g0+k0g1).reshape(*k1grad.data.shape)
            k1block.values[...] *= k0block.values[...]

    return kernel


def keys_intersection(t1, t2):
    keys1 = {tuple(key) for key in t1.keys}
    keys2 = {tuple(key) for key in t2.keys}
    return np.array(sorted(keys1 & keys2, key=lambda x: x[::-1]))


def compute_prediction(kernel, weights, averages=None):

    cblocks = []
    keys = keys_intersection(kernel, weights)
    tm_labels = equistore.Labels(weights.keys.names, keys)
    for (l, q) in tm_labels:
        kblock = kernel.block(spherical_harmonics_l=l, species_center=q)
        wblock = weights.block(spherical_harmonics_l=l, element=q)
        values = np.einsum('amMr,rMn->amn', kblock.values, wblock.values)
        if averages and l==0:
            values += averages.block(element=q).values
        cblock = equistore.TensorBlock(values=values, samples=kblock.samples,
                                       components=wblock.components, properties=wblock.properties)

        if kblock.has_gradient('positions'):
            kgrad = kblock.gradient('positions')
            data = np.einsum('admMr,rMn->admn', kgrad.data, wblock.values)
            cblock.add_gradient(parameter='positions', data=data,
                                samples=kgrad.samples, components=kgrad.components[0:1]+cblock.components)
        cblocks.append(cblock)

    coeffs = equistore.TensorMap(keys=tm_labels, blocks=cblocks)
    return coeffs
