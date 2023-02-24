ISO=1e-5  # density isovalue
NP=230    # number of points: 6  14 26 38 50 74 86 110 146 170 194 230 266 302 350 434 590 770 974
          # 1202 1454 1730 2030 2354 2702 3074 3470 3890 4334 4802 5294 5810
MOL=caffeine.xyz

# prepare input files
./prepare.py ${MOL} ${MOL}.coeff.dat ${ISO} ${NP}
# compute grid
./q230224 ccPVQZ_JKFIT.in ${MOL}.{in,grid.out}
# extract the grid
sed -n '/<<< grid:/,/^$/p' ${MOL}.grid.out | tail -n +2 > ${MOL}.grid.dat
# change the task in the input
sed -i s/isodensity/esp/ ${MOL}.in
# compute the esp
./q230224 ccPVQZ_JKFIT.in ${MOL}.{in,esp.out}
# TODO
# extract the esp and plot it
