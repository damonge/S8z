from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os


data_folder = '/mnt/bluewhale/damonge/S8z_data/derived_products'
output_folder = '/mnt/bluewhale/gravityls_3/S8z/Cls/Planck'
####
des_folder_gcl = 'des_clustering'
des_mask = 'mask_ns4096.fits'
des_nside = 4096
des_data_folder = os.path.join(data_folder, des_folder_gcl)
des_mask_path = os.path.join(des_data_folder, des_mask)

# Read mask
des_mask = hp.read_map(des_mask_path, verbose=False)

fsky = np.mean(des_mask)  # Use des_mask for binning as we had
d_ell = int(1./fsky)
b = nmt.NmtBin(des_nside,nlb=d_ell)

####
planck_folder = 'planck_lensing'
planck_data_folder = os.path.join(data_folder, planck_folder)
fname = os.path.join(planck_data_folder, 'mask_ns4096.fits')
planck_mask = hp.read_map(fname)
fname = os.path.join(planck_data_folder, 'map_kappa_ns4096.fits')
planck_map_kappa = hp.read_map(fname)
####
f0 = nmt.NmtField(planck_mask, [planck_map_kappa])

ws = nmt.NmtWorkspace()
fname = os.path.join(output_folder, 'Planck_w00_55.dat')
ws.compute_coupling_matrix(f0, f0, b)
ws.write_to(fname)

cls = ws.decouple_cell(nmt.compute_coupled_cell(f0, f0)).reshape((1, 1, -1))

np.savez(os.path.join(output_folder, "cl_TT_Planck_with_Planck_w00_55_workspace.npz"),
         l=b.get_effective_ells(), cls=cls)
