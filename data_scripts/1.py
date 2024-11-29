import sys
import metatensor

x = metatensor.load(sys.argv[1])

xx = []
for key, block in x:
    xx.append(block.copy())

tm1 = ['spherical_harmonics_l', 'species_center']
tm2 = ['spherical_harmonics_l1', 'spherical_harmonics_l2', 'species_center1', 'species_center2']


try:
    keys = metatensor.Labels(names=tm1, values=x.keys.asarray())
except:
    keys = metatensor.Labels(names=tm2, values=x.keys.asarray())

y = metatensor.TensorMap(blocks=xx, keys=keys)
metatensor.save(sys.argv[1], y)
