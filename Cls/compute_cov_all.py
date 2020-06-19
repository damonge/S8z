#!/usr/bin/python

from __future__ import print_function
from scipy.interpolate import interp1d
import os
import sys
import pymaster as nmt
import numpy as np
import healpy as hp
import common as co

# pylint: disable=C0103,C0111

##############################################################################
##############################################################################
nside = 4096
wltype = 'metacal'
obsdir = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_metacal_new'
outdir = os.path.join(obsdir, 'new_fiducial_cov')

##############################################################################
################ Read Cls ###############
##############################################################################
############# Load theory cl matrix

bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
index_B = [6, 8, 10, 12]
th_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons'
th_ell, th_cls_all = co.load_cls_all_array_from_files(th_folder, bins, index_B)

############# Load obs. cl matrix

lbpw_obs_cls_all = np.load(os.path.join(obsdir, 'cl_all_with_noise.npz'))
lbpw, obs_cls_all_wn = lbpw_obs_cls_all['l'], lbpw_obs_cls_all['cls']

############### Load DES noises

desgc_nls_arr = np.load(os.path.join(obsdir, 'des_w_cl_shot_noise_ns{}.npz'.format(nside)))['cls']
dessh_nls_arr = np.load(os.path.join(obsdir, 'des_sh_{}_noise_ns{}.npz'.format(wltype, nside)))['cls']

############## Add noise
for i, nls_i in enumerate(desgc_nls_arr):
    th_cls_all[i, i] += interp1d(lbpw, nls_i, bounds_error=False,
                                 fill_value=(nls_i[0], nls_i[-1]))(th_ell)

for i, nls_i in enumerate(dessh_nls_arr):
    ish = len(desgc_nls_arr) + 2 * i
    th_cls_all[ish, ish] += interp1d(lbpw, nls_i[0, 0], bounds_error=False,
                                     fill_value=(nls_i[0, 0, 0], nls_i[0, 0, -1]))(th_ell)
    th_cls_all[ish+1, ish+1] += interp1d(lbpw, nls_i[1, 1], bounds_error=False,
                                     fill_value=(nls_i[1, 1, 0], nls_i[1, 1, -1]))(th_ell)
############## Use observed Planck's Cls for auto-correlation
th_cls_all[-1, -1] = interp1d(lbpw, obs_cls_all_wn[-1, -1], bounds_error=False,
                                 fill_value=(obs_cls_all_wn[-1, -1, 0], obs_cls_all_wn[-1, -1, -1]))(th_ell)


np.savez_compressed(outdir + '/th_cls_all_with_noise.npz', ell=th_ell, cls=th_cls_all)

##############################################################################
############################### Load masks ###################################
##############################################################################
def load_masks():
    masks = []
    #Read mask
    fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
    masks.append(hp.read_map(fname, verbose=False))

    for ibin in range(4):
        fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/map_{}_bin{}_w_ns{}.fits'.format(wltype, ibin, nside)
        masks.append(hp.read_map(fname, verbose=False))

    fname = '/mnt/extraspace/damonge/S8z_data/derived_products/planck_lensing/mask_ns{}.fits'.format(nside)'
    masks.append(hp.read_map(fname))
    masks = np.array(masks)

    return masks


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
        fname = os.path.join(obsdir, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        ws.read_from(fname)
        return ws

def compute_covariance_full(clTh, nbpw, nbins, maps_bins, maps_spins, maps_masks, masks):

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

        ##### Couple Cl
        wa1b1 = get_workspace_from_spins_masks(s_a1, s_b1, m_a1, m_b1)
        cla1b1 = wa1b1.couple_cell(cla1b1)
        #
        if m_b2 != m_b1:
            wa1b2 = get_workspace_from_spins_masks(s_a1, s_b2, m_a1, m_b2)
        else:
            wa1b2 = wa1b1
        cla1b2 = wa1b2.couple_cell(cla1b2)
        #
        if m_a2 != m_a1:
            wa2b1 = get_workspace_from_spins_masks(s_a2, s_b1, m_a2, m_b1)
        else:
            wa2b1 = wa1b1
        cla2b1 = wa2b1.couple_cell(cla2b1)
        #
        if m_b2 != m_b1:
            wa2b2 = get_workspace_from_spins_masks(s_a2, s_b2, m_a2, m_b2)
        else:
            wa2b2 = wa2b1
        cla2b2 = wa2b2.couple_cell(cla2b2)
        #####

        #### Weight the Cls
        cla1b1 = cla1b1 / np.mean(masks[m_a1] * masks[m_b1])
        cla1b2 = cla1b2 / np.mean(masks[m_a1] * masks[m_b2])
        cla2b1 = cla2b1 / np.mean(masks[m_a2] * masks[m_b1])
        cla2b2 = cla2b2 / np.mean(masks[m_a2] * masks[m_b2])
        ####

        wa = get_workspace_from_spins_masks(s_a1, s_a2, m_a1, m_a2)
        if (m_a1 == m_b1) and (m_a2 == m_b2):
            wb = wa
        else:
            wb = get_workspace_from_spins_masks(s_b1, s_b2, m_b1, m_b2)

        fname_cw = os.path.join(obsdir, 'cw{}{}{}{}.dat'.format(*cov_masks[i]))
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

        tracer_names = [co.get_tracer_name(ibin) for ibin in cov_bins[i]]
        fname_new = os.path.join(outdir, 'cov_{}_{}_{}_{}.npz'.format(*tracer_names))
        #  TT -> 0; TE -> 0;  EE -> 0
        np.savez_compressed(fname_new, cov.reshape((nbpw, na1 * na2, nbpw, nb1 * nb2))[:, 0, :, 0])


    # Loop through cov_indices, use below algorithm and compute the Cov
    # Check wich one has been computed, store/save it and remove them form cov_indices

###################

nbins = 5 + 4 + 1
nmaps = 5 + 8 + 1
maps_bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
maps_masks = [0] * 5 + [1, 1] + [2, 2] + [3, 3] + [4, 4] + [5]
maps_spins = [0] * 5 + [2, 2] * 4 + [0]
masks = load_masks()
compute_covariance_full(th_cls_all, lbpw.size, nbins, maps_bins, maps_spins, maps_masks, masks)
