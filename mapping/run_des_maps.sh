#!/bin/bash

nside=4096
nrandom=100

# Clustering
echo python3 des_clustering.py
python3 des_clustering.py

# Shear
#  - Loop over shape catalogs
for cat_name in metacal im3shape
do
    #  - Loop over redshift bins
    for bin in 0 1 2 3
    do
	echo python3 des_shear.py ${cat_name} ${nside} ${bin}
	python3 des_shear.py ${cat_name} ${nside} ${bin}
	#  - Loop over random rotations (needed for noise bias estimation)
	for ((irot=0;irot<${nrandom};irot++)); do
	    echo python3 des_shear ${cat_name} ${nside} ${bin} do_rotate ${irot}
	    python3 des_shear ${cat_name} ${nside} ${bin} do_rotate ${irot}
	done
    done
done
