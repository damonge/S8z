#!/bin/bash

#nside=512
#mem=0.5
#nc=12
nside=4096
mem=5
nc=24

icov=0
ia=0
#{0..3}
for ba1 in 0
do
    for ba2 in {0..3}
    do
	if [[ ${ba2} -lt ${ba1} ]]; then
	    continue
	fi

	ib=0
	for bb1 in {0..3}
	do
	    for bb2 in {0..3}
	    do
		if [[ ${bb2} -lt ${bb1} ]]; then
		    continue
		fi
		if [[ ${ib} -lt ${ia} ]]; then
		    ((ib++))
		    continue
		fi
		comment="cv_${ba1}${ba2}_${bb1}${bb2}_${ia}_${ib}_${icov}_ns${nside}"
		pyexec="addqueue -c ${comment} -n 1x${nc} -s -q berg -m ${mem} /usr/bin/python3"
		#comm="${pyexec} covs_metacal.py --bin-a1 ${ba1} --bin-a2 ${ba2} --bin-b1 ${bb1} --bin-b2 ${bb2} --nside ${nside} --n-iter 0"
		comm="${pyexec} covs_metacal_dev.py --bin-a1 ${ba1} --bin-a2 ${ba2} --bin-b1 ${bb1} --bin-b2 ${bb2} --nside ${nside} --n-iter 0 --full-noise"
		
		echo ${comment}
		${comm}
		((ib++))
		((icov++))
		#exit
	    done
	done
	((ia++))
    done
done


#usage: covs_metacal.py [-h] [--bin-a1 BIN_A1] [--bin-a2 BIN_A2]
#                     [--bin-b1 BIN_B1] [--bin-b2 BIN_B2] [--nside NSIDE]
#                     [--n-iter N_ITER] [--recompute-mcm]
#
#optional arguments:
#  -h, --help       show this help message and exit
#  --bin-a1 BIN_A1  Bin number
#  --bin-a2 BIN_A2  Bin number
#  --bin-b1 BIN_B1  Bin number
#  --bin-b2 BIN_B2  Bin number
#  --nside NSIDE    Nside
#  --n-iter N_ITER  n_iter
#  --recompute-mcm  Recompute MCM even if it exists?
