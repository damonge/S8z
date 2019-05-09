#!/usr/bin/python

from __future__ import print_function
from scipy.interpolate import interp1d
import os
import pymaster as nmt
import numpy as np
import healpy as hp

# pylint: disable=C0103,C0111

##############################################################################
##############################################################################
output_folder = '/mnt/bluewhale/gravityls_3/S8z/Cls'

data_folder = '/mnt/bluewhale/damonge/S8z_data/derived_products'
des_folder_gcl = 'des_clustering'
des_mask = 'mask_ns4096.fits'
des_nside = 4096
des_bins = 5

des_data_folder = os.path.join(data_folder, des_folder_gcl)

des_mask_path = os.path.join(des_data_folder, des_mask)
des_th_cls_path = '/mnt/bluewhale/evam/S8z/'

##############################################################################
########### Read mask ###########
##############################################################################
des_mask = hp.read_map(des_mask_path, verbose=False)

##############################################################################
########### Read map ###########
##############################################################################
# Read one map (gg) (as they all share the same mask)
map_file = os.path.join(des_data_folder, 'map_counts_w_bin0_ns4096.fits')  # Same mask, just one field needed
des_mapi = hp.read_map(map_file)

des_N_mean = des_mapi.sum() / des_mask.sum()
des_mapi_dg = des_mapi / (des_N_mean * des_mask) - 1
des_mapi_dg[np.isnan(des_mapi_dg)] = 0.  # This is the map for Cl's

##############################################################################
############## Set up binning scheme ##############
##########################################nvim####################################
fsky = np.mean(des_mask)
d_ell = int(1./fsky)
b = nmt.NmtBin(des_nside, nlb=d_ell)

##############################################################################
################ Read Th. cls ###############
##############################################################################

##############################################################################
def load_thcls(th_outdir):
    cls_arr = []
    for i in range(des_bins):
        for j in range(i, des_bins):
            fname = os.path.join(th_outdir, 'DES_Cls_lmax3xNside_{}_{}.txt'.format(i, j))
            if not os.path.isfile(fname): #spin0-spin0
                raise ValueError('Missing workspace: ', fname)

            cls_arr.append(np.loadtxt(fname, usecols=1))

    ell = np.loadtxt(fname, usecols=0)

    return ell, np.array(cls_arr)
##############################################################################

th_ell, des_th_cls_arr = load_thcls(des_th_cls_path)

des_th_cl00_matrix = np.empty((des_bins, des_bins, len(th_ell)),
                              dtype=des_th_cls_arr[0].dtype)
i, j = np.triu_indices(des_bins)
des_th_cl00_matrix[i, j] = des_th_cls_arr
des_th_cl00_matrix[j, i] = des_th_cls_arr

##############################################################################
################# Add shot noise Cls to th's ones  #################
##############################################################################
fname = os.path.join(output_folder, "des_w_cl_shot_noise_ns4096.npz")
if not os.path.isfile(fname):
    raise ValueError('Missing shot noise: ', fname)

des_Nls_file = np.load(fname)
des_Nls_ell = des_Nls_file['l']
des_Nls_arr = des_Nls_file['cls']

for i, nls in enumerate(des_Nls_arr):
    des_th_cl00_matrix[i, i] += interp1d(des_Nls_ell, nls, bounds_error=False,
                                         fill_value=(nls[0], nls[-1]))(th_ell)


##############################################################################
##############################################################################
####################### NaMaster-thing part ##################################
##############################################################################
##############################################################################

des_f0 = nmt.NmtField(des_mask, [des_mapi_dg])
f0 = des_f0

##############################################################################
###############  Load already computed workspace   ###############
##############################################################################
fname = os.path.join(output_folder, 'des_w00_ns4096.dat')
w00 = nmt.NmtWorkspace()
if not os.path.isfile(fname): #spin0-spin0
    w00.compute_coupling_matrix(f0, f0, b)  # raise ValueError('Missing workspace: ', fname)
    w00.write_to(fname)
else:
    w00.read_from(fname)

##############################################################################
################### Compute covariance coupling coefficients #################
##############################################################################
fname = os.path.join(output_folder, 'des_cw_ns4096.npz')
cw = nmt.NmtCovarianceWorkspace()
if not os.path.isfile(fname):
    cw.compute_coupling_coefficients(f0, f0)
    cw.write_to(fname)
else:
    cw.read_from(fname)

##############################################################################
#################  Compute coupling cov matrix for bins i, j #################
##############################################################################
for i in range(5):
    clt1t1 = des_th_cl00_matrix[i, i]
    for j in range(i, 5):
        clt2t2 = des_th_cl00_matrix[j, j]
        clt1t2 = des_th_cl00_matrix[i, j]

        fname = os.path.join(output_folder, 'des_c0000_{}{}.npz'.format(i, j))
        if os.path.isfile(fname):
            continue

        c0000 = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [clt1t1], [clt1t2], [clt1t2], [clt2t2], w00)
        np.savez_compressed(fname, c0000)
