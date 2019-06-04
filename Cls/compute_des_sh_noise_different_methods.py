#!/usr/bin/python
from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import sys

# pylint: disable=C0103

# Compute DES shear noise

def des_sh_nls_rot_gal(des_mask_gwl, des_opm_mean, des_data_folder_gwl, output_folder, b, nbins=4, galaxy_treatment='metacal'):
    des_wl_noise_file = os.path.join(output_folder, "des_sh_{}_rot0-10_noise_ns4096.npz".format(galaxy_treatment))
    if os.path.isfile(des_wl_noise_file):
        N_wl = np.load(des_wl_noise_file)['cls']
        return N_wl

    N_wl = []
    for ibin in range(nbins):
        rotated_cls = []
        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w22_{}{}.dat'.format(1 + ibin, 1 + ibin))
        ws.read_from(fname)

        for irot in range(10):
            map_file_e1 = os.path.join(des_data_folder_gwl, 'map_{}_bin{}_rot{}_counts_e1_ns4096.fits'.format(galaxy_treatment, ibin, irot))
            map_file_e2 = os.path.join(des_data_folder_gwl, 'map_{}_bin{}_rot{}_counts_e2_ns4096.fits'.format(galaxy_treatment, ibin, irot))

            map_we1 = hp.read_map(map_file_e1)
            map_we2 = hp.read_map(map_file_e2)

            map_e1 = (map_we1/des_mask_gwl[ibin] - (map_we1.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e2 = (map_we2/des_mask_gwl[ibin] - (map_we2.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e1[np.isnan(map_e1)] = 0.
            map_e2[np.isnan(map_e2)] = 0.

            sq = map_e1
            su = -map_e2
            f = nmt.NmtField(des_mask_gwl[ibin], [sq, su])

            cls = ws.decouple_cell(nmt.compute_coupled_cell(f, f)).reshape((2, 2, -1))
            rotated_cls.append(cls)

        N_wl.append(np.mean(rotated_cls, axis=0))

    np.savez(des_wl_noise_file,
             l=b.get_effective_ells(), cls=N_wl)

    return N_wl

def des_sh_nls_rot_map(des_mask_gwl, des_opm_mean, des_data_folder_gwl, output_folder, b, nbins=4, galaxy_treatment='metacal', Nrot=10):
    des_wl_noise_file = os.path.join(output_folder, "des_sh_{}_rot_maps0-{}_noise_ns4096.npz".format(galaxy_treatment, Nrot))
    if os.path.isfile(des_wl_noise_file):
        N_wl = np.load(des_wl_noise_file)['cls']
        return N_wl


    rotated_cls = np.zeros((nbins, 2, 2, b.get_n_bands()))
    for ibin in range(nbins):
        fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_e1_ns4096.fits'.format(i))
        map_we1_orig = hp.read_map(fname)
        fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_e2_ns4096.fits'.format(i))
        map_we2_orig = hp.read_map(fname)

        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w22_{}{}.dat'.format(1 + ibin, 1 + ibin))
        ws.read_from(fname)
        print('Reading {}'.format(fname))

        for irot in range(Nrot):
            rnumbers = np.random.uniform(0, 2*np.pi, size=map_we1_orig.size)
            map_we1 = map_we1_orig * np.cos(rnumbers)
            map_we2 = map_we2_orig * np.sin(rnumbers)

            map_e1 = (map_we1/des_mask_gwl[ibin] - (map_we1.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e2 = (map_we2/des_mask_gwl[ibin] - (map_we2.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e1[np.isnan(map_e1)] = 0.
            map_e2[np.isnan(map_e2)] = 0.

            sq = map_e1
            su = -map_e2
            f = nmt.NmtField(des_mask_gwl[ibin], [sq, su])

            print('Computing cls. ibin = {}, irot = {}'.format(ibin, irot))
            print('Any is nan? {}'.format(np.isnan([sq, su]).any(axis=1)))
            cls = ws.decouple_cell(nmt.compute_coupled_cell(f, f)).reshape((2, 2, -1))

            rotated_cls[ibin] += cls

    N_wl = rotated_cls / Nrot

    np.savez(des_wl_noise_file,
             l=b.get_effective_ells(), cls=N_wl)

    return N_wl


if __name__ == '__main__':
    output_folder = '/mnt/bluewhale/gravityls_3/S8z/Cls/all_together_linear_binning'
    data_folder = '/mnt/bluewhale/damonge/S8z_data/derived_products'
    nside = 4096

    # ######
    # ells = np.arange(3 * nside)
    # ells_lim_bpw= np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])
    # bpws = np.zeros(ells.shape)
    # weights = np.zeros(ells.shape)

    # li = 0
    # for i, lf in enumerate(ells_lim_bpw[1:]):
    #     # lf += 1
    #     bpws[li : lf] = i
    #     weights[li : lf] += 1./weights[li : lf].size
    #     li = lf

    # b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)
    # ######

    des_folder_gcl = 'des_clustering'
    des_mask = 'mask_ns4096.fits'

    des_data_folder = os.path.join(data_folder, des_folder_gcl)

    des_mask_path = os.path.join(des_data_folder, des_mask)

    # Read mask
    # mask_lss = hp.ud_grade(hp.read_map(des_mask_path, verbose=False), nside_out=2048)
    des_mask = hp.read_map(des_mask_path, verbose=False)

    #Set up binning scheme
    fsky = np.mean(des_mask)
    d_ell = int(1./fsky)
    b = nmt.NmtBin(nside,nlb=d_ell)

    ########

    des_folder_gwl = 'des_shear'
    des_data_folder_gwl = os.path.join(data_folder, des_folder_gwl)
    des_mask_gwl = []
    des_maps_wopm = []
    for i in range(4):
        fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_w_ns4096.fits'.format(i))
        des_mask_gwl.append(hp.read_map(fname))
        fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_opm_ns4096.fits'.format(i))
        des_maps_wopm.append(hp.read_map(fname))

    des_mask_gwl = np.array(des_mask_gwl)
    des_maps_wopm = np.array(des_maps_wopm)
    des_opm_mean = des_maps_wopm.sum(axis=1)/des_mask_gwl.sum(axis=1)


    des_sh_nls_rot_gal(des_mask_gwl, des_opm_mean, des_data_folder_gwl, output_folder, b)
    des_sh_nls_rot_map(des_mask_gwl, des_opm_mean, des_data_folder_gwl, output_folder, b)
