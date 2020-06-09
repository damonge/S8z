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
    /usr/bin/python3 des_shear.py --catalog ${cat_name} --nside ${nside} --bin-number ${bin} --rotate --seed ${irot}
done
