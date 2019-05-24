#!/usr/bin/python

from __future__ import print_function
from scipy.interpolate import interp1d
import os
import sys
import pymaster as nmt
import numpy as np
import healpy as hp

# pylint: disable=C0103,C0111

##############################################################################
##############################################################################
output_folder = '/mnt/bluewhale/gravityls_3/S8z/Cls/all_together'
des_th_cls_path = '/mnt/bluewhale/evam/S8z/'

des_maps = des_bins = 5
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

##############################################################################
##############################################################################
##############################################################################

def get_nelems_spin(spin):
    if spin == 0:
        return 1
    if spin == 2:
        return 2

def get_workspace_from_spins_masks(spin1, spin2, mask1, mask2):
        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        ws.read_from(fname)
        return ws

def compute_covariance_full(clTh, nbins, maps_bins, maps_spins, maps_masks):

    nmaps = clTh.shape[0]
    fname_cw_old = ''

    cl_indices = []
    cl_spins = []
    cl_bins = []
    cl_masks = []
    for i in range(nmaps):
        si = maps_spins[i]
        for j in range(i, nmaps):
            sj = maps_spins[j]
            cl_indices.append([i, j])
            cl_spins.append([si, sj])
            cl_bins.append([maps_bins[i], maps_bins[j]])
            cl_masks.append([maps_masks[i], maps_masks[j]])

    cov_indices = []
    cov_spins = []
    cov_bins = []
    cov_masks = []
    for i, clij in enumerate(cl_indices):
        for j, clkl in enumerate(cl_indices[i:]):
            cov_indices.append(cl_indices[i] + cl_indices[i + j])
            cov_spins.append(cl_spins[i] + cl_spins[i + j])
            cov_bins.append(cl_bins[i] + cl_bins[i + j])
            cov_masks.append(cl_masks[i] + cl_masks[i + j])


    cov_indices = np.array(cov_indices)
    cov_spins = np.array(cov_spins)
    cov_bins = np.array(cov_bins)
    cov_masks = np.array(cov_masks)

    for i, indices in enumerate(cov_indices):
        s_a1, s_a2, s_b1, s_b2 = cov_spins[i]
        m_a1, m_a2, m_b1, m_b2 = cov_masks[i]

        na1 = get_nelems_spin(s_a1)
        na2 = get_nelems_spin(s_a2)
        nb1 = get_nelems_spin(s_b1)
        nb2 = get_nelems_spin(s_b2)

        bin_a1, bin_a2, bin_b1, bin_b2 = cov_bins[i]

        fname = os.path.join(output_folder, 'cov_c{}{}{}{}_{}{}{}{}.npz'.format(*cov_spins[i], *cov_masks[i]))
        if os.path.isfile(fname):
            continue

        ibin_a1 = np.where(maps_bins == bin_a1)[0][0] + int(s_a1 / 2)
        ibin_a2 = np.where(maps_bins == bin_a2)[0][0] + int(s_a2 / 2)
        ibin_b1 = np.where(maps_bins == bin_b1)[0][0] + int(s_b1 / 2)
        ibin_b2 = np.where(maps_bins == bin_b2)[0][0] + int(s_b2 / 2)

        cla1b1 = np.concatenate(clTh[ibin_a1 : ibin_a1 + na1, ibin_b1 : ibin_b1 + nb1])
        cla1b2 = np.concatenate(clTh[ibin_a1 : ibin_a1 + na1, ibin_b2 : ibin_b2 + nb2])
        cla2b1 = np.concatenate(clTh[ibin_a2 : ibin_a2 + na2, ibin_b1 : ibin_b1 + nb1])
        cla2b2 = np.concatenate(clTh[ibin_a2 : ibin_a2 + na2, ibin_b2 : ibin_b2 + nb2])

        wa = get_workspace_from_spins_masks(s_a1, s_a2, m_a1, m_a2)
        wb = get_workspace_from_spins_masks(s_b1, s_b2, m_b1, m_b2)

        fname_cw = os.path.join(output_folder, 'cw{}{}{}{}.dat'.format(m_a1, m_a2, m_b1, m_b2))
        if fname_cw != fname_cw_old:
            cw = nmt.NmtCovarianceWorkspace()
            cw.read_from(fname)
            fname_cw_old = fname_cw

        # cla1b1_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b1 : ibin_b1 + nb1])
        # cla1b2_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b2 : ibin_b2 + nb2])
        # cla2b1_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b1 : ibin_b1 + nb1])
        # cla2b2_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b2 : ibin_b2 + nb2])

        # print(np.concatenate(cla1b1))
        # print(np.concatenate(cla1b2))
        # print(np.concatenate(cla2b1))
        # print(np.concatenate(cla2b2))

        # print('Computing ', fname)
        # print('spins: ', s_a1, s_a2, s_b1, s_b2)
        # print('cla1b1', (s_a1, s_b1), cla1b1.shape, ibin_a1, ibin_a1 + na1, ibin_b1, ibin_b1 + nb1, cla1b1_label)
        # print('cla1b2', (s_a1, s_b2), cla1b2.shape, ibin_a1, ibin_a1 + na1, ibin_b2, ibin_b2 + nb2, cla1b2_label)
        # print('cla2b1', (s_a2, s_b1), cla2b1.shape, ibin_a2, ibin_a2 + na2, ibin_b1, ibin_b1 + nb1, cla2b1_label)
        # print('cla2b2', (s_a2, s_b2), cla2b2.shape, ibin_a2, ibin_a2 + na2, ibin_b2, ibin_b2 + nb2, cla2b2_label)

        cov = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                                      cla1b1, cla1b2, cla2b1, cla2b2,
                                      wa, wb)

        np.savez_compressed(fname, cov)


    # Loop through cov_indices, use below algorithm and compute the Cov
    # Check wich one has been computed, store/save it and remove them form cov_indices

nbins = 5 + 4 + 1
nmaps = 5 + 8 + 1
maps_bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
maps_masks = [0] * 5 + [1, 1] + [2, 2] + [3, 3] + [4, 4] + [5]
maps_spins = [0] * 5 + [2, 2] * 4 + [0]
compute_covariance_full(Clth, nbins, maps_bins, maps_spins, maps_masks)
