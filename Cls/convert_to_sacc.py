#!/usr/bin/python

import numpy as np
import argparse
import common as co
import pymaster as nmt
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
# Create tracers (interested only on shear)
##############################################################################
s = sacc.Sacc()

ntracers_gc = 5
ntracers_wl = 4
ntracers_cv = 1

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

s_noise = s.copy()
##############################################################################
# Define bandpower windows
##############################################################################

b = co.get_NmtBin(nside)
nells_nobin = 3 * nside
ells_nobin = np.arange(nells_nobin)
nbpw = b.get_n_bands()

window_ar = np.zeros((nbpw, nells_nobin))
for i in range(nbpw):
    window_ar[i, b.get_ell_list(i)] = b.get_weight_list(i)

ws_bpw = {}
for i, tr1 in enumerate(tracer_names):
    for j, tr2 in enumerate(tracer_names[i:], i):
        ws = nmt.NmtWorkspace()
        fname = os.path.join(clsdir, 'w22_{}{}.dat'.format(i+1, j+1))
        ws.read_from(fname)
        ws_bpw[tr1+tr2] = ws.get_bandpower_windows()

ws = None

##############################################################################
# Add Noise to SACC s_noise instance
##############################################################################
fname = os.path.join(clsdir, "des_sh_{}_noise_ns{}.npz".format(wltype, nside))
clfile = np.load(fname)
lbpw = clfile['l']
if not np.all(lbpw == b.get_effective_ells()):
    raise 'Bandpowers from NmtBin and des_sh_{}_noise_ns{}.npz file do NOT coincide'.format(wltype, nside)
nls = clfile['cls']

for i in range(ntracers_wl):
    cl_type = 'cl_ee'
    tr1 = tr2 = 'wl{}'.format(i)
    wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[tr1+tr2][0, :, 0, :].T)
    s_noise.add_ell_cl(cl_type, tr1, tr2,
                 lbpw, nls[i, 0, 0],
                 window=wins)
    cl_type = 'cl_eb'
    tr1 = tr2 = 'wl{}'.format(i)
    wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[tr1+tr2][1, :, 1, :].T)
    s_noise.add_ell_cl(cl_type, tr1, tr2,
                 lbpw, nls[i, 0, 0]*0,
                 window=wins)
    cl_type = 'cl_be'
    tr1 = tr2 = 'wl{}'.format(i)
    wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[tr1+tr2][2, :, 2, :].T)
    s_noise.add_ell_cl(cl_type, tr1, tr2,
                 lbpw, nls[i, 0, 0]*0,
                 window=wins)
    cl_type = 'cl_bb'
    tr1 = tr2 = 'wl{}'.format(i)
    wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[tr1+tr2][3, :, 3, :].T)
    s_noise.add_ell_cl(cl_type, tr1, tr2,
                 lbpw, nls[i, 1, 1],
                 window=wins)

fname = os.path.join(clsdir, "DESwl_nls.fits")
s_noise.save_fits(fname, overwrite=args.overwrite)
##############################################################################

##############################################################################
# Continue with detected Cls
##############################################################################
# Add Cls
##############################################################################

fname = os.path.join(clsdir, "cl_all_no_noise.npz")
clfile = np.load(fname)
lbpw = clfile['l']
if not np.all(lbpw == b.get_effective_ells()):
    raise 'Bandpowers from NmtBin and cl_all_no_noise file do NOT coincide'
cl_matrix = clfile['cls']

field_names = []
for tr in tracer_names:
    field_names.append(tr + '_e')
    field_names.append(tr + '_b')

cl_tracers = []
for i, fn1 in enumerate(field_names):
    for j, fn2 in enumerate(field_names[i:], i):
        cl_type = 'cl_{}{}'.format(fn1[-1], fn2[-1])
        tr1 = fn1.split('_')[0]
        tr2 = fn2.split('_')[0]
        if fn1[-1] + fn2[-1] == 'ee':
            wsix = 0
        elif fn1[-1] + fn2[-1] == 'eb':
            wsix = 1
        elif fn1[-1] + fn2[-1] == 'be':
            wsix = 2
        else:
            wsix = 3
        wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[tr1+tr2][wsix, :, wsix, :].T)
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
covmat = -1 * np.ones((ncls * nbpw, ncls * nbpw))
for i, trs1 in enumerate(cl_tracers):
    for j, trs2 in enumerate(cl_tracers[i:], i):
        # Get bin number
        b1 = int(trs1[0].split('_')[0][-1]) + 5
        b2 = int(trs1[1].split('_')[0][-1]) + 5
        b3 = int(trs2[0].split('_')[0][-1]) + 5
        b4 = int(trs2[1].split('_')[0][-1]) + 5
        # Get cov matrix indexes for given modes
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
fname = os.path.join(clsdir, "DESwl.fits")
s.save_fits(fname, overwrite=args.overwrite)
