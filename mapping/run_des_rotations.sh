#!/bin/bash

nside=$1
nrot=$2
irot0=$3
cat_name=$4
bin=$5

echo $nside
echo $nrot
echo $cat_name

for((irot=${irot0};irot<${nrot};irot++)); do
    /usr/bin/python3 des_shear.py ${cat_name} ${nside} ${bin} do_rotate ${irot}
done
