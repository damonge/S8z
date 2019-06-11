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
outdir = '/mnt/bluewhale/gravityls_3/S8z/Cls/all_together'

##############################################################################
################ Define functions to read Cls ###############
##############################################################################
def load_thcls(th_outdir, file_prefix, nmaps):
    cls_arr = []
    for i in range(nmaps):
        for j in range(i, nmaps):
            fname = os.path.join(th_outdir, file_prefix + '_{}_{}.txt'.format(i, j))
            if not os.path.isfile(fname): #spin0-spin0
                raise ValueError('Missing workspace: ', fname)

            cls_arr.append(np.loadtxt(fname, usecols=1))

    ell = np.loadtxt(fname, usecols=0)

    return ell, np.array(cls_arr)

def load_thcls_gk(nmaps_g, nmaps_k):
    th_outdir = '/mnt/bluewhale/evam/S8z/Clsgk/'
    file_prefix = 'DES_Cls_gk_lmax3xNside'
    cls_arr = []
    for i in range(nmaps_g):
        for j in range(nmaps_k):
            fname = os.path.join(th_outdir, file_prefix + '_{}_{}.txt'.format(i, j))
            if not os.path.isfile(fname): #spin0-spin0
                raise ValueError('Missing workspace: ', fname)

            cls_arr.append(np.loadtxt(fname, usecols=1))

    ell = np.loadtxt(fname, usecols=0)

    return ell, np.array(cls_arr)

def load_thcls_Planck():
    fdir = '/mnt/bluewhale/evam/S8z/ClsPlanck/'
    cls_arr = []
    for i in range(5):
        fname = os.path.join(fdir, 'DESPlanck_Cls_gk_lmax3xNside_{}.txt'.format(i))
        cls_arr.append(np.loadtxt(fname, usecols=1))
    for i in range(4):
        fname = os.path.join(fdir, 'DESPlanck_Cls_kk_lmax3xNside_{}.txt'.format(i))
        cls_arr.append(np.loadtxt(fname, usecols=1))
        cls_arr.append(cls_arr[-1] * 0)

    fname = os.path.join(fdir, 'Planck_Cls_kk_lmax3xNside.txt')
    cls_arr.append(np.loadtxt(fname, usecols=1))
    ell = np.loadtxt(fname, usecols=0)

    return ell, np.array(cls_arr)

def load_cls_all_matrix_th():
    # All th_ell are the same
    th_outdir = '/mnt/bluewhale/evam/S8z/Clsgg/'
    th_ell, Clsgg_ar = load_thcls(th_outdir, 'DES_Cls_lmax3xNside', 5)

    th_outdir = '/mnt/bluewhale/evam/S8z/Clskk/'
    th_ell, Clskk_ar = load_thcls(th_outdir, 'DES_Cls_kk_lmax3xNside', 4)

    th_outdir = '/mnt/bluewhale/evam/S8z/Clsgk/'
    th_ell, Clsgk_ar = load_thcls_gk(5, 4)

    th_outdir = '/mnt/bluewhale/evam/S8z/ClsPlanck/'
    th_ell, ClsPlanck_ar = load_thcls_Planck()

    # Checked that all EE's are the same as in the array.
    Clskk_full_mat = np.zeros((8, 8, th_ell.shape[0]))
    i, j = np.triu_indices(4)
    Clskk_full_mat[::2, ::2][i, j] = Clskk_ar
    Clskk_full_mat[::2, ::2][j, i] = Clskk_ar
    i, j = np.triu_indices(8)
    Clskk_ar_full = Clskk_full_mat[i, j]

    th_cls_all = np.zeros((14, 14, th_ell.shape[0]))

    i, j = np.triu_indices(5)
    th_cls_all[:5, :5][i, j] = Clsgg_ar

    i, j = np.triu_indices(8)
    th_cls_all[5:-1, 5:-1][i, j] = Clskk_ar_full

    th_cls_all[:, -1] = ClsPlanck_ar

    for i in range(5):
        th_cls_all[i, 5:-1:2] = Clsgk_ar[i * 4 : (i + 1) * 4]

    i, j = np.triu_indices(14)
    th_cls_all_ar = th_cls_all[i, j]
    th_cls_all[j, i] = th_cls_all_ar

    return th_ell, th_cls_all

##############################################################################
################ Read Cls ###############
##############################################################################
############# Load theory cl matrix

th_ell, th_cls_all = load_cls_all_matrix_th()

############# Load obs. cl matrix

lbpw_obs_cls_all = np.load(os.path.join(outdir, 'cl_all_with_noise.npz'))
lbpw, obs_cls_all_wn = lbpw_obs_cls_all['l'], lbpw_obs_cls_all['cls']

############### Load DES noises

desgc_nls_arr = np.load(os.path.join(outdir, 'des_w_cl_shot_noise_ns4096.npz'))['cls']
dessh_nls_arr = np.load(os.path.join(outdir, 'des_sh_metacal_rot0-10_noise_ns4096.npz'))['cls']

############## Add noise
for i, nls_i in enumerate(desgc_nls_arr):
    th_cls_all[i, i] += interp1d(lbpw, nls_i, bounds_error=False,
                                 fill_value=(nls_i[0], nls_i[-1]))(th_ell)

for i, nls_i in enumerate(dessh_nls_arr):
    ish = len(desgc_nls_arr) + 2 * i
    th_cls_all[ish : ish + 2, ish : ish + 2] += interp1d(lbpw, nls_i, bounds_error=False,
                                                     fill_value=(nls_i[:, :, 0], nls_i[:, :, -1]))(th_ell)
############## Use observed Planck's Cls
th_cls_all[:, -1] = interp1d(lbpw, obs_cls_all_wn[:, -1], bounds_error=False,
                                 fill_value=(obs_cls_all_wn[:, -1, 0], obs_cls_all_wn[:, -1, -1]))(th_ell)
th_cls_all[-1, :] = interp1d(lbpw, obs_cls_all_wn[-1, :], bounds_error=False,
                             fill_value=(obs_cls_all_wn[-1, :, 0], obs_cls_all_wn[-1, :, -1]))(th_ell)


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
        fname = os.path.join(outdir, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        ws.read_from(fname)
        return ws

def get_tracer_name(ibin):
    if ibin in np.arange(5):
        name = 'DESgc{}'.format(ibin)
    elif ibin in np.arange(5, 9):
        name = 'DESwl{}'.format(ibin-5)
    elif ibin == 9:
        name = 'PLAcv'

    return name

def compute_covariance_full(clTh, nbpw, nbins, maps_bins, maps_spins, maps_masks):

    nmaps = len(maps_bins)
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

        fname = os.path.join(outdir, 'cov_s{}{}{}{}_b{}{}{}{}.npz'.format(*cov_spins[i], *cov_bins[i]))
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

        fname_cw = os.path.join(outdir, 'cw{}{}{}{}.dat'.format(*cov_masks[i]))
        if fname_cw != fname_cw_old:
            cw = nmt.NmtCovarianceWorkspace()
            cw.read_from(fname_cw)
            fname_cw_old = fname_cw

        # cla1b1_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b1 : ibin_b1 + nb1])
        # cla1b2_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b2 : ibin_b2 + nb2])
        # cla2b1_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b1 : ibin_b1 + nb1])
        # cla2b2_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b2 : ibin_b2 + nb2])

        # print(np.concatenate(cla1b1))
        # print(np.concatenate(cla1b2))
        # print(np.concatenate(cla2b1))
        # print(np.concatenate(cla2b2))

        print('Computing {}'.format(fname))
        # print('spins: ', s_a1, s_a2, s_b1, s_b2)
        # print('cla1b1', (s_a1, s_b1), cla1b1.shape, ibin_a1, ibin_a1 + na1, ibin_b1, ibin_b1 + nb1, cla1b1_label)
        # print('cla1b2', (s_a1, s_b2), cla1b2.shape, ibin_a1, ibin_a1 + na1, ibin_b2, ibin_b2 + nb2, cla1b2_label)
        # print('cla2b1', (s_a2, s_b1), cla2b1.shape, ibin_a2, ibin_a2 + na2, ibin_b1, ibin_b1 + nb1, cla2b1_label)
        # print('cla2b2', (s_a2, s_b2), cla2b2.shape, ibin_a2, ibin_a2 + na2, ibin_b2, ibin_b2 + nb2, cla2b2_label)

        cov = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                                      cla1b1, cla1b2, cla2b1, cla2b2,
                                      wa, wb)

        np.savez_compressed(fname, cov)

        tracer_names = [get_tracer_name(ibin) for ibin in cov_bins[i]]
        fname_new = os.path.join(outdir, 'cov_{}_{}_{}_{}.npz'.format(*tracer_names))
        #  TT -> 0; TE -> 0;  EE -> 0
        np.savez_compressed(fname_new, cov.reshape((nbpw, na1 * na2, nbpw, nb1 * nb2))[:, 0, :, 0])


    # Loop through cov_indices, use below algorithm and compute the Cov
    # Check wich one has been computed, store/save it and remove them form cov_indices

nbins = 5 + 4 + 1
nmaps = 5 + 8 + 1
maps_bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
maps_masks = [0] * 5 + [1, 1] + [2, 2] + [3, 3] + [4, 4] + [5]
maps_spins = [0] * 5 + [2, 2] * 4 + [0]
compute_covariance_full(th_cls_all, lbpw.size, nbins, maps_bins, maps_spins, maps_masks)
