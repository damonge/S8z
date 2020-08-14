#!/usr/bin/python

from __future__ import print_function
import argparse
import os
import sys
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import numpy as np

# pylint: disable=C0103

##############################################################################

parser = argparse.ArgumentParser(description="Compute PSFi-ei Cls")
parser.add_argument('--nside',  default=4096, type=int,
                    help='HEALPix nside param')
parser.add_argument('--wltype', default='metacal', type=str,
                    help='DES weak lensing shear measurement algorithm (metacal or im3shape)')
parser.add_argument('--plot', default=False, action='store_true',
                    help='Set if you want to produce plots')
parser.add_argument('--outdir', default='', type=str, help='Path to compute PSFi-ei Cls')

args = parser.parse_args()

##############################################################################
data_folder = '/mnt/extraspace/damonge/S8z_data/derived_products'

wltype = args.wltype
nside = args.nside

# Output folder
if args.outdir:
    output_folder = args.outdir
elif nside == 4096:
    output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_new'.format(wltype)
else:
    output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_{}_new'.format(wltype, nside)
os.makedirs(output_folder, exist_ok=True)
##############################################################################
############################ DES Weak lensing ################################
##############################################################################

folder_gwl = 'des_shear'
data_folder_gwl = os.path.join(data_folder, folder_gwl)
masks_gwl = []
maps_e1 = []
maps_e2 = []
maps_psfe1 = []
maps_psfe2 = []
for i in range(4):
    fname = os.path.join(data_folder_gwl, 'sums_{}_bin{}.npz'.format(wltype, i))
    sums = np.load(fname)
    fname = os.path.join(data_folder_gwl, 'map_{}_bin{}_w_ns{}.fits'.format(wltype, i, nside))
    mask_gwl = hp.read_map(fname)
    fname = os.path.join(data_folder_gwl, 'map_{}_bin{}_we1_ns{}.fits'.format(wltype, i, nside))
    map_we1 = hp.read_map(fname)
    fname = os.path.join(data_folder_gwl, 'map_{}_bin{}_we2_ns{}.fits'.format(wltype, i, nside))
    map_we2 = hp.read_map(fname)
    #
    fname = os.path.join(data_folder_gwl, 'map_{}_bin{}_wpsfe1_ns{}.fits'.format(wltype, i, nside))
    map_wpsfe1 = hp.read_map(fname)
    fname = os.path.join(data_folder_gwl, 'map_{}_bin{}_wpsfe2_ns{}.fits'.format(wltype, i, nside))
    map_wpsfe2 = hp.read_map(fname)

    mask_good = mask_gwl > 0.
    mask_gwl[~mask_good] = 0.
    opm_mean = sums['wopm'] / sums['w']

    map_e1 = np.zeros_like(map_we1)
    map_e2 = np.zeros_like(map_we2)
    map_e1[mask_good] = (map_we1[mask_good]/mask_gwl[mask_good]) / opm_mean
    map_e2[mask_good] = (map_we2[mask_good]/mask_gwl[mask_good]) / opm_mean

    map_psfe1 = np.zeros_like(map_wpsfe1)
    map_psfe2 = np.zeros_like(map_wpsfe2)
    map_psfe1[mask_good] = map_wpsfe1[mask_good]/mask_gwl[mask_good]
    map_psfe2[mask_good] = map_wpsfe2[mask_good]/mask_gwl[mask_good]

    masks_gwl.append(mask_gwl)
    maps_e1.append(map_e1)
    maps_e2.append(map_e2)
    maps_psfe1.append(map_psfe1)
    maps_psfe2.append(map_psfe2)

masks_gwl = np.array(masks_gwl)
maps_e1 = np.array(maps_e1)
maps_e2 = np.array(maps_e2)
maps_psfe1 = np.array(maps_psfe1)
maps_psfe2 = np.array(maps_psfe2)

##############################################################################
############################ Compute fields ##################################
##############################################################################

fields_psf = []
fields_e = []

for i in range(maps_e1.shape[0]):
    sq = maps_e1[i]
    su = -maps_e2[i]
    fields_e.append(nmt.NmtField(masks_gwl[i], [sq, su]))

    sq = maps_psfe1[i]
    su = -maps_psfe2[i]
    fields_psf.append(nmt.NmtField(masks_gwl[i], [sq, su]))

##############################################################################
############################### Compute Cls ##################################
##############################################################################

fname = os.path.join(output_folder, 'cls_psfi-ei.npz')
fname2 = os.path.join(output_folder, 'cls_psfi-psfi.npz')
if not os.path.isfile(fname):
    spin1 = spin2 = 2
    dof1 = dof2 = 2

    cls = []
    cls2 = []
    for i in range(4):
        mask1 = mask2 = i + 1
        ws = nmt.NmtWorkspace()
        ws_fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        ws.read_from(ws_fname)

        f1 = fields_psf[i]
        f2 = fields_e[i]
        cls.append(ws.decouple_cell(nmt.compute_coupled_cell(f1, f2)).reshape((dof1, dof2, -1)))
        cls2.append(ws.decouple_cell(nmt.compute_coupled_cell(f1, f1)).reshape((dof1, dof2, -1)))

    lbpw = np.loadtxt(os.path.join(output_folder, 'l_bpw.txt'))
    cls = np.array(cls)
    np.savez(fname, l=lbpw, cls=cls)
    np.savez(fname2, l=lbpw, cls=cls2)

##############################################################################
############################ Compute covmat ##################################
##############################################################################

fname_cov = os.path.join(output_folder, 'cov_psfi-ei.npz')
if not os.path.isfile(fname_cov):
    lbpw = np.loadtxt(os.path.join(output_folder, 'l_bpw.txt'))
    spin1 = spin2 = 2
    covar_ar = []
    for i in range(4):
        mask1 = mask2 = i+1

        # Load worskpace
        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        ws.read_from(fname)

        # Load covariance workspace
        cw = nmt.NmtCovarianceWorkspace()
        fname = os.path.join(output_folder, 'cw{0}{0}{0}{0}.dat'.format(mask1))
        if not os.path.isfile(fname): # mask2-mask2-mask2-mask2
            print("Computing", fname)
            f2 = fields_e[i]

            cw.compute_coupling_coefficients(f2, f2, f2, f2)
            cw.write_to(fname)
        else:
            cw.read_from(fname)

        # Prepare Cls
        # PSF - PSF
        f1 = fields_psf[i]
        cla1b1 = nmt.compute_coupled_cell(f1, f1)
        cla1b1 /= np.mean(masks_gwl[i]**2)

        # PSF - wl
        cla1b2 = cla2b1 = np.zeros_like(cla1b1)

        # wl1 - wl1
        fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESwl{}_DESwl{}.npz'.format(i, i)
        clwlwl_ee = np.load(fname)['cls'][:3*nside]
        fname = os.path.join(output_folder, 'des_sh_{}_noise_ns{}.npz'.format(wltype, nside))
        nls = np.load(fname)

        cla2b2 = np.array([clwlwl_ee, 0*clwlwl_ee, 0*clwlwl_ee, 0*clwlwl_ee])
        cla2b2 = ws.couple_cell(cla2b2)
        cla2b2 += nls['cls_raw'][i].reshape(cla2b2.shape)
        cla2b2 /= np.mean(masks_gwl[i]**2)


        covar = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                        cla1b1,  # PP -> EE, EB, BE, BB
                                        cla1b2,  # Pe -> EE, EB, BE, BB
                                        cla2b1,  # eP -> EE, EB, BE, BB
                                        cla2b2,  # ee -> EE, EB, BE, BB
                                        ws, wb=ws).reshape([lbpw.size, 4,
                                                            lbpw.size, 4])

        covar_ar.append(covar)
        np.savez(fname_cov, l=lbpw, cov=covar_ar)
