#!/usr/bin/python
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

(o, args) = parser.parse_args()
##############################################################################
# Create outdir
if not os.path.exists(o.outdir):
    raise ValueError("outdir does not exist:".format(o.outdir))
# Set files prefix
prefix_out = os.path.join(o.outdir, 'run_gc0gc0')
##############################################################################

##############################################################################
################## Covariance from Simulations ###############################
##############################################################################
cl_ar = np.load(run_path + '_cl_0001-0100.npz')['cl00']
C = np.cov(cl_for_C)
fname = prefix_out + '_covSims_0001-0100.npz' # sims_suffix
np.savez_compressed(fname, C)

##############################################################################
###################### Covariance from Theory ################################
#############################################################################

s_a1 = s_a2 = s_b1 = s_b2 = 0
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc0_DESgc0.npz'
cla1b1 = cla1b2 = cla2b1 = cla2b2 = np.load(fname)['cls']

w00=nmt.NmtWorkspace();
w00.read_from('/mnt/extraspace/gravityls_3/S8z/Cls/all_together_2048/w00_00.dat')
wa = wb = w00

cw = nmt.NmtCovarianceWorkspace()
cw.read_from('/mnt/extraspace/gravityls_3/S8z/Cls/all_together_2048/cw0000.dat')

C = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                            cla1b1, cla1b2, cla2b1, cla2b2,
                            wa, wb)
fname = prefix_out + '_covTh.npz'
np.savez_compressed(fname, C)
