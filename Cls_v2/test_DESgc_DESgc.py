#!/usr/bin/python

import pymaster as nmt
import pyccl as ccl
import numpy as np
import healpy as hp
import os
import argparse

parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
parser.add_argument('ibin', type=str, help='Tomobin')
parser.add_argument('--no_nl_correction', default=False, action='store_true', help='Remove the nl_cp correction factor')
args = parser.parse_args()

ibin = args.ibin
corr = not args.no_nl_correction
corr_suffix = ''
if not corr:
    corr_suffix = '_nocorr'

##### Config #####
spin = 0
bias = 1.41
threshold = 0.5

path2w = f'/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/map_counts_w_bin{ibin}_ns4096.fits.gz'
path2w2 = f'/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/map_counts_w2_bin{ibin}_ns4096.fits.gz'
path2mask = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns4096.fits'
path2dndz = f'/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/dndz_bin{ibin}.txt'
dndz_cols = [1, 3]

bpw_edges = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])
nside = 4096

output = '/mnt/extraspace/gravityls_3/prueba_DESgc/'
os.makedirs(output, exist_ok=True)
##### END Config #####

# Bin #
bpw_edges = bpw_edges[bpw_edges <= 3 * nside] # 3*nside == ells[-1] + 1
if 3 * nside not in bpw_edges: # Exhaust lmax --> gives same result as previous method, but adds 1 bpw (not for 4096)
    bpw_edges = np.append(bpw_edges, 3*nside)
b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
#######
# DN/DZ #
z, dn = np.loadtxt(path2dndz, usecols=dndz_cols, unpack=True)
#########
# Maps #
w = hp.read_map(path2w)
w2 = hp.read_map(path2w2)
#######
# Mask #
mask = hp.read_map(path2mask)
goodpix = mask > threshold
mask[~goodpix] = 0
########
# Overdensity map #
Nmean = np.sum(w[goodpix]) / np.sum(mask)
dg = np.zeros_like(mask)
dg[goodpix] = w[goodpix] / (Nmean * mask[goodpix]) - 1
###################

# Field #
f = nmt.NmtField(mask, [dg], n_iter=0)
wsp = nmt.NmtWorkspace()
path2wsp= output + 'w.fits'
if os.path.isfile(path2wsp):
    wsp.read_from(path2wsp)
else:
    wsp.compute_coupling_matrix(f, f, b)
    wsp.write_to(path2wsp)
#########

#########
# Nl #
npix = hp.nside2npix(nside)
Nmean_srad = Nmean / (4 * np.pi) * npix
N_ell = mask.sum() / npix / Nmean_srad
if corr:
    correction = w2[goodpix].sum() / w[goodpix].sum()
else:
    correction = 1
N_ell *= correction # New_noise
nl_cp = N_ell * np.ones((1, 3 * nside))
# Cl #
cl_cp = nmt.compute_coupled_cell(f, f)
cl_nl = wsp.decouple_cell(cl_cp)
nl = wsp.decouple_cell(nl_cp)
cl = cl_nl - nl
np.savez_compressed(output + f'cl_DESgc{ibin}_DESgc{ibin}{corr_suffix}.npz', cl=cl, nl=nl, cl_cp=cl_cp, nl_cp=nl_cp)
######

#####
# Cov #
cwsp = nmt.NmtCovarianceWorkspace()
path2cwsp = output + 'cw.fits'
if os.path.isfile(path2cwsp):
    cwsp.read_from(path2cwsp)
else:
    cwsp.compute_coupling_coefficients(f, f)
    cwsp.write_to(path2cwsp)

# Clfid #
clfid = np.load(f'/mnt/extraspace/gravityls_3/S8z/Cls_2/4096_asDavid/fiducial/DESgc_DESgc/cl_DESgc{ibin}_DESgc{ibin}.npz')['cl']
clfid_cp = wsp.couple_cell(clfid)

cla1b1 = cla1b2 = cla2b1 = cla2b2 = (clfid_cp + nl_cp) / np.mean(mask * mask)
cov = nmt.gaussian_covariance(cwsp, 0, 0, 0, 0, cla1b1, cla1b2, cla2b1, cla2b2, wa=wsp, wb=wsp)
np.savez_compressed(output + f'cov_DESgc{ibin}_DESgc{ibin}_DESgc{ibin}_DESgc{ibin}{corr_suffix}.npz', cov)
