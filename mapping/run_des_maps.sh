#!/bin/bash

#nside=4096
nside=512
nrot=20
irot0=10

# Clustering
echo python3 des_clustering.py ${nside}
addqueue -q cmb -m 12 -n 1 /usr/bin/python3 des_clustering.py ${nside}

# Shear
#  - Loop over shape catalogs
for cat_name in metacal im3shape
do
    #  - Loop over redshift bins
    for bin in 0 1 2 3
    do
	echo python3 des_shear.py ${cat_name} ${nside} ${bin}
	addqueue -q cmb -m 12 -n 1 /usr/bin/python3 des_shear.py ${cat_name} ${nside} ${bin}
	#  - Loop over random rotations (needed for noise bias estimation)
	addqueue -q cmb -m 12 -n 1 ./run_des_rotations.sh ${nside} ${nrot} ${irot0} ${cat_name} ${bin}
    done
done
