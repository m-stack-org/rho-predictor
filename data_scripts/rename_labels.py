import sys
import numpy as np
import metatensor

x = metatensor.load(sys.argv[1])

xx = []
for key, block in x.items():
    xx.append(block.copy())

tm1 = ['o3_lambda', 'center_type']
tm2 = ['o3_lambda1', 'o3_lambda2', 'center_type1', 'center_type2']

try:
    keys = metatensor.Labels(names=tm1, values=np.array(x.keys))
except:
    keys = metatensor.Labels(names=tm2, values=np.array(x.keys))

y = metatensor.TensorMap(blocks=xx, keys=keys)
metatensor.save(sys.argv[1], y)
