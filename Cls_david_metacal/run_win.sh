#!/bin/bash

#nside=512
#mem=0.5
#nside=2048
#mem=1
nside=4096
mem=2
nc=12

for bin1 in {0..3}
do
    for bin2 in {0..3}
    do
	if [[ ${bin2} -lt ${bin1} ]]; then
	    continue
	fi
	comment="ss_${bin1}_${bin2}_ns${nside}"
	pyexec="addqueue -c ${comment} -n 1x${nc} -s -q berg -m ${mem} /usr/bin/python3"
	comm="${pyexec} windows_metacal.py --bin-number ${bin1} --bin-number-2 ${bin2} --nside ${nside}"
	echo ${comment}
	${comm}
    done
done
