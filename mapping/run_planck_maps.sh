#!/bin/bash

nside=4096

echo python3 planck_lensing.py ${nside}
addqueue -q cmb -m 12 -n 1 /usr/bin/python3 planck_lensing.py ${nside}
