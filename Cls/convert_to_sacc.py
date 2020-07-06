#!/usr/bin/python

import numpy as np
import argparse
import common as co
import sacc
import os

##############################################################################
##############################################################################
parser = argparse.ArgumentParser(description="Compute PSFi-ei Cls")
parser.add_argument('--nside',  default=4096, type=int,
                    help='HEALPix nside param')
parser.add_argument('--wltype', default='metacal', type=str,
                    help='DES weak lensing shear measurement algorithm (metacal or im3shape)')
parser.add_argument('--overwrite', default=False, type=bool,
                    help='Overwrite output file')

args = parser.parse_args()
##############################################################################

wltype = args.wltype
nside = args.nside

# Output folder
if nside == 4096:
    clsdir = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_new'.format(wltype)
else:
    clsdir = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_{}_new'.format(wltype, nside)

covdir = os.path.join(clsdir, 'cov_new_fiducial')
obspath = '/mnt/extraspace/damonge/S8z_data/derived_products/'
obspath_wl = os.path.join(obspath, 'des_shear')
##############################################################################
# Define bandpower windows
##############################################################################

b = co.get_NmtBin(nside)
nells_nobin = 3 * nside
nbpw = b.get_n_bands()

window_ar = np.zeros(nbpw, nells_nobin)
for i in range(nbpw):
    window_ar[i, b.get_ell_list(i)] = b.get_weight_list(i)

wins = sacc.BandpowerWindow(nells_nobin, window_ar.T)

##############################################################################
# Create tracers (interested only on shear)
##############################################################################
s = sacc.Sacc()

ntracers_gc = 5
ntracers_wl = 4
# ntracers_cv = 1

tracer_names = []
for i in range(ntracers_wl):
    fname = os.path.join(obspath_wl, 'dndz_{}_bin{}.txt'.format(wltype, i))
    dndz = np.loadtxt(fname)
    z=dndz[1]
    nz=dndz[3]
    name = 'wl{}'.format(i)

    s.add_tracer('NZ', name,
                 quantity='galaxy_shear',
                 spin=2,
                 z=z,
                 nz=nz)
    tracer_names.append(name)

##############################################################################
# Add Cls
##############################################################################

fname = os.path.join(clsdir, "cl_all_no_noise")
clfile = np.load(fname)
lbpw = clfile['l']
if not (lbpw == b.get_effective_ells()):
    raise 'Bandpowers from NmtBin and cl_all_no_noise file do NOT coincide'
cl_matrix = clfile['cls']

field_names = []
for tr in tracer_names:
    field_names.append(tr + '_e')
    field_names.append(tr + '_b')

cl_tracers = []
for i, fn1 in enumerate(field_names):
    for j, fn2 in enumerate(field_names[i:, ], i):
        cl_type = 'cl_{}{}'.format(fn1[-1], fn2[-1])
        tr1 = fn1.split('_')[0]
        tr2 = fn2.split('_')[0]
        s.add_ell_cl(cl_type, tr1, tr2,
                     lbpw, cl_matrix[i + ntracers_gc, j + ntracers_gc],
                     window=wins)
        cl_tracers.append([fn1, fn2])

##############################################################################
# Add covmat
##############################################################################
clmodes_index = {'ee': 0,
                 'eb': 1,
                 'be': 2,
                 'bb': 3}

ncls = len(cl_tracers)
covmat = -1 * np.ones(ncls, ncls)
for i, trs1 in enumerate(cl_tracers):
    for j, trs2 in enumerate(cl_tracers[i:], i):
        b1 = trs1[0].split('_')[0][-1]
        b2 = trs1[1].split('_')[0][-1]
        b3 = trs2[0].split('_')[0][-1]
        b4 = trs2[1].split('_')[0][-1]
        index1 = clmodes_index[trs1[0][-1] + trs1[1][-1]]
        index2 = clmodes_index[trs2[0][-1] + trs2[1][-1]]

        fname = os.path.join(covdir, 'cov_s2222_b{}{}{}{}.npz'.format(b1, b2,
                                                                      b3, b4))
        cov = np.load(fname)['arr_0'].reshape(nbpw, 4, nbpw, 4)[:, index1, :, index2]  # Not efficient!

        covmat[i * nbpw : (i+1) * nbpw, j * nbpw : (j+1) * nbpw] = cov
        covmat[j * nbpw : (j+1) * nbpw, i * nbpw : (i+1) * nbpw] = cov.T

s.add_covariance(covmat)
##############################################################################
# Save
##############################################################################
s.save_fits("DES_wl.fits", overwrite=args.overwrite)
