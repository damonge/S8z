#!/bin/bash

#nside=512
#mem=0.5
#nc=12
nside=4096
mem=5
nc=24
irot_0=100
nrot=1000

for bin1 in {0..3}
do
    for bin2 in {0..3}
    do
	if [[ ${bin2} -lt ${bin1} ]]; then
	    continue
	fi
	comment="ss_${bin1}_${bin2}_ns${nside}"
	pyexec="addqueue -c ${comment} -n 1x${nc} -s -q cmb -m ${mem} /usr/bin/python3"
	comm="${pyexec} cls_metacal.py --bin-number ${bin1} --bin-number-2 ${bin2} --nside ${nside} --n-iter 0"
	echo ${comment}
	echo ${comm}
    done
done

for bin1 in {0..3}
do
    #Rotations
    comment="rots_${bin1}_ns${nside}"
    pyexec="addqueue -c ${comment} -n 1x${nc} -s -q berg -m ${mem} /usr/bin/python3"
    comm="${pyexec} cls_metacal.py --bin-number ${bin1} --nside ${nside} --n-iter 0 --irot-0 ${irot_0} --irot-f ${nrot}"
    echo ${comment}
    ${comm}
    #PSF-x
    comment="psfX_${bin1}_ns${nside}"
    pyexec="addqueue -c ${comment} -n 1x${nc} -s -q cmb -m ${mem} /usr/bin/python3"
    comm="${pyexec} cls_metacal.py --bin-number ${bin1} --nside ${nside} --n-iter 0 --is-psf-x"
    echo ${comment}
    echo ${comm}
    #PSF-a
    comment="psfA_${bin1}_ns${nside}"
    pyexec="addqueue -c ${comment} -n 1x${nc} -s -q cmb -m ${mem} /usr/bin/python3"
    comm="${pyexec} cls_metacal.py --bin-number ${bin1} --nside ${nside} --n-iter 0 --is-psf-a"
    echo ${comment}
    echo ${comm}
done

#usage: cls_metacal.py [-h] [--bin-number BIN_NUMBER] [--nside NSIDE]
#                    [--n-iter N_ITER] [--bin-number-2 BIN_NUMBER_2]
#                    [--is-psf-x] [--is-psf-a] [--irot IROT] [--recompute-mcm]
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --bin-number BIN_NUMBER
#                        Bin number
#  --nside NSIDE         Nside
#  --n-iter N_ITER       n_iter
#  --bin-number-2 BIN_NUMBER_2
#                        Bin number
#  --is-psf-x            Compute psf cross-correlation
#  --is-psf-a            Compute psf auto-correlation
#  --irot IROT           Rotation number
#  --recompute-mcm       Recompute MCM even if it exists?
