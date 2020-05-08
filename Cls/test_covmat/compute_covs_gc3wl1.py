#!/usr/bin/python
from optparse import OptionParser
from scipy.interpolate import interp1d
import pymaster as nmt
import numpy as np
import os
import sys

# pylint: disable=C0103

# prefix = 'run_sph_2b_same_mask'

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--outdir',dest='outdir',default='./sims',type=str,
                  help='Output directory')
parser.add_option('--nside', dest='nside', default=4096, type=int,
                  help='HEALPix nside param')

(o, args) = parser.parse_args()
##############################################################################
# Create outdir
if not os.path.exists(o.outdir):
    raise ValueError("outdir does not exist:".format(o.outdir))
# Set files prefix
prefix_out = os.path.join(o.outdir, 'run_gc3wl1')
# Set root path of observations
obs_path = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together'
if o.nside != 4096:
    obs_path += '_{}'.format(o.nside)
##############################################################################

##############################################################################
################## Covariance from Simulations ###############################
##############################################################################
cl_ar = np.load(prefix_out + '_clsims_0001-0100.npz')['cl00'][:, 0, :]
C = np.cov(cl_ar.T)
fname = prefix_out + '_covSims_0001-0100.npz' # sims_suffix
np.savez_compressed(fname, C)

##############################################################################
###################### Covariance from Theory ################################
#############################################################################

#Read input power spectra
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc3_DESwl1.npz'
fid_data = np.load(fname)
l, cltt = fid_data['ells'], fid_data['cls']

fname = os.path.join(obs_path, 'des_w_cl_shot_noise_ns{}.npz'.format(o.nside))
nls = np.load(fname)
nltt = interp1d(nls['l'],  nls['cls'][0], bounds_error=False,
                fill_value=(nls['cls'][0, 0], nls['cls'][0, -1]))(l)
# nside = C.shape[0]
# cltt=cltt[:3*nside]
# nltt=nltt[:3*nside]

# a1 = gc3, b1 = wl1, a2 = gc3, b2 = wl1
# all combinations are gc3-wl1
cla1b1 = cla2b2 = cla1b2 = cla2b1 =(cltt+nltt).reshape(1, -1)
s_a1 = s_a2 = 0
s_b1 = s_b2 = 2
###

w00=nmt.NmtWorkspace();
w00.read_from(os.path.join(obs_path, 'w02_02.dat')
wa = wb = w00

cw = nmt.NmtCovarianceWorkspace()
cw.read_from(os.path.join(obs_path, 'cw0202.dat')

C = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                            cla1b1, cla1b2, cla2b1, cla2b2,
                            wa, wb)
fname = prefix_out + '_covTh.npz'
np.savez_compressed(fname, C)
