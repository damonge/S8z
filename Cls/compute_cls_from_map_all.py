from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

# pylint: disable=C0103

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
# parser.add_option('--nside', dest='nside', default=512, type=int,
#                   help='HEALPix nside param')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')

(o, args) = parser.parse_args()

##############################################################################
##############################################################################
##############################################################################

output_folder = '/mnt/bluewhale/gravityls_3/S8z/Cls/all_together'
data_folder = '/mnt/bluewhale/damonge/S8z_data/derived_products'
# nside = 4096

##############################################################################
############################ DES Clustering ##################################
##############################################################################

des_folder_gcl = 'des_clustering'
des_mask = 'mask_ns4096.fits'
des_nside = 4096

des_data_folder = os.path.join(data_folder, des_folder_gcl)

des_mask_path = os.path.join(des_data_folder, des_mask)

# Read mask
# mask_lss = hp.ud_grade(hp.read_map(des_mask_path, verbose=False), nside_out=2048)
des_mask = hp.read_map(des_mask_path, verbose=False)
# Read maps (gg)
nmaps = 5
des_maps = []
for i in range(nmaps):
    map_file = os.path.join(des_data_folder, 'map_counts_w_bin{}_ns4096.fits'.format(i))
    des_maps.append(hp.read_map(map_file))
des_maps = np.array(des_maps)

des_N_mean = des_maps.sum(axis=1) / des_mask.sum()
des_maps_dg = des_maps / (des_N_mean[:, None] * des_mask) - 1
des_maps_dg[np.isnan(des_maps_dg)] = 0.

# ###### Test ######
#
# for i, mapi in enumerate(des_maps_dg):
#     check = des_maps[i] / (des_N_mean * des_mask) - 1
#     check[np.isnan(check)] = 0
#
#     print(np.all(mapi == check))
#
# ###### Test ######

##############################################################################
############################ DES Weak lensing ################################
##############################################################################

des_folder_gwl = 'des_shear'
des_data_folder_gwl = os.path.join(data_folder, des_folder_gwl)
des_mask_gwl = []
des_maps_we1 = []
des_maps_we2 = []
des_maps_wopm = []
for i in range(4):
    fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_w_ns4096.fits'.format(i))
    des_mask_gwl.append(hp.read_map(fname))
    fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_e1_ns4096.fits'.format(i))
    des_maps_we1.append(hp.read_map(fname))
    fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_e2_ns4096.fits'.format(i))
    des_maps_we2.append(hp.read_map(fname))
    fname = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_counts_opm_ns4096.fits'.format(i))
    des_maps_wopm.append(hp.read_map(fname))

des_mask_gwl = np.array(des_mask_gwl)
des_maps_we1 = np.array(des_maps_we1)
des_maps_we2 = np.array(des_maps_we2)
des_maps_wopm = np.array(des_maps_wopm)

des_opm_mean = des_maps_wopm.sum(axis=1)/des_mask_gwl.sum(axis=1)

des_maps_e1 = (des_maps_we1/des_mask_gwl - (des_maps_we1.sum(axis=1)/des_mask_gwl.sum(axis=1))[:,None]) / des_opm_mean[:, None]
des_maps_e2 = (des_maps_we2/des_mask_gwl - (des_maps_we2.sum(axis=1)/des_mask_gwl.sum(axis=1))[:,None]) / des_opm_mean[:, None]
des_maps_e1[np.isnan(des_maps_e1)] = 0.
des_maps_e2[np.isnan(des_maps_e2)] = 0.

###### Test ######

# for i, mapi in enumerate(des_maps_e1):
#     check = (des_maps_we1[i]/des_mask_gwl[i] - des_maps_we1[i].sum()/des_mask_gwl[i].sum()) / des_opm_mean[i]
#     check[np.isnan(check)] = 0
#
#     print(np.all(mapi == check))
#
# ###### Test ######

##############################################################################
############################# Planck Lensing #################################
##############################################################################

planck_folder = 'planck_lensing'
planck_data_folder = os.path.join(data_folder, planck_folder)
fname = os.path.join(planck_data_folder, 'mask_ns4096.fits')
planck_mask = hp.read_map(fname)
fname = os.path.join(planck_data_folder, 'map_kappa_ns4096.fits')
planck_map_kappa = hp.read_map(fname)

##############################################################################
########################## Putting all together ##############################
##############################################################################

maps = np.empty((len(des_maps_dg) + len(des_maps_e1) * 2 + 1, planck_map_kappa.shape[0]))
maps[:len(des_maps_dg)] = des_maps_dg
maps[-1] = planck_map_kappa

ix = len(des_maps_dg)
for i, map_e1 in enumerate(des_maps_e1):
    maps[ix] = map_e1
    maps[ix+1] = des_maps_e2[i]
    ix += 2

# print(maps.shape)
# print(np.all(maps[-1] == planck_map_kappa))

masks = [0] * len(des_maps_dg) + [1, 1] + [2, 2] + [3, 3] + [4, 4] + [5]

masks_dic = {0: des_mask,
             5: planck_mask}

for i, maski in enumerate(des_mask_gwl):
    masks_dic.update({i+1: maski})

spins = [0] * len(des_maps_dg) + [2] * 2 * len(des_maps_e1) + [0]

#Set up binning scheme
fsky = np.mean(des_mask)  # Use des_mask for binning as we had
d_ell = int(1./fsky)
b = nmt.NmtBin(des_nside,nlb=d_ell)

##############################################################################
############################# NaMaster stuff #################################
##############################################################################

##############################################################################
# Generate fields for each map
##############################################################################
fields = []
for mapi in des_maps_dg:
    fields.append(nmt.NmtField(des_mask, [mapi]))

for i in range(des_maps_e1.shape[0]):
    sq = des_maps_e1[i]
    su = - des_maps_e2[i]
    f = nmt.NmtField(des_mask_gwl[i], [sq, su])
    fields += [f, f]

fields.append(nmt.NmtField(planck_mask, [planck_map_kappa]))

##############################################################################
# Generate workspaces
##############################################################################
workspaces_fnames_ar = []  # Use fnames to save space

for i in range(len(maps)):
    for j in range(i, len(maps)):
        spin1 = spins[i]
        spin2 = spins[j]
        mask1 = masks[i]
        mask2 = masks[j]
        fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
        if not os.path.isfile(fname):
            w = nmt.NmtWorkspace()
            f1 = fields[i]
            f2 = fields[j]
            w.compute_coupling_matrix(f1, f2, b)
            w.write_to(fname)

        workspaces_fnames_ar.append(fname)

##############################################################################
# Compute Cls
##############################################################################
def get_nelems_spin(spin):
    if spin == 0:
        return 1
    if spin == 2:
        return 2

# i_triu, j_triu = np.triu_indices(len(maps))

cls_noise_file = os.path.join(output_folder, "cl_all_with_noise.npz")
if os.path.isfile(cls_noise_file):
    cl_matrix = np.load(cls_noise_file)['cls']
else:
    cl_matrix = np.empty((len(maps), len(maps), b.get_n_bands()))

    index1 = 0
    dof1 = dof2 = 0
    for c1, f1 in enumerate(fields):
        index2 = index1
        if dof1 == 2:
            dof1 = 0
            continue
        spin1 = spins[c1]
        mask1 = masks[c1]
        dof1 = get_nelems_spin(spin1)
        for c2, f2 in enumerate(fields[c1:]):
            c2 += c1
            if dof2 == 2:
                dof2 = 0
                continue
            spin2 = spins[c2]
            mask2 = masks[c2]
            dof2 = get_nelems_spin(spin2)
            ws = nmt.NmtWorkspace()
            fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(spin1, spin2, mask1, mask2))
            ws.read_from(fname)

            cls = ws.decouple_cell(nmt.compute_coupled_cell(f1, f2)).reshape((dof1, dof2, -1))

            # from matplotlib import pyplot as plt
            # cls_true = (f['cls'] + f['nls'])[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)
            # print(cls_true.shape)
            # print(cls.reshape(dof1 * dof2, -1).shape)
            # for cli_true, cli in zip(cls_true, cls.reshape(dof1 * dof2, -1)):
            #     print(cli)
            #     plt.suptitle("{}, {}".format(dof1, dof2))
            #     plt.loglog(l, cli_true, b.get_effective_ells(), cli, 'o')
            #     plt.show()
            #     plt.close()

            cl_matrix[index1 : index1 + dof1, index2 : index2 + dof2] = cls

             # from matplotlib import pyplot as plt
             # for cli_true, cli in zip(cls_true,
             #                          cl_ar[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)):
             #     plt.suptitle("{}, {}".format(dof1, dof2))
             #     plt.loglog(l, cli_true, b.get_effective_ells(), cli, 'o')
             #     plt.show()
             #     plt.close()

            index2 += dof2
        index1 += dof1

    i, j = np.triu_indices(len(maps))
    cl_arr = cl_matrix[i, j]
    cl_matrix[j, i] = cl_arr

    np.savez(cls_noise_file,
             l=b.get_effective_ells(), cls=cl_matrix)

# ##############################################################################
# # Compute Noise
# ##############################################################################

# Compute DES galaxy clustering noise
des_gc_noise_file = os.path.join(output_folder, "des_w_cl_shot_noise_ns4096.npz")
if os.path.isfile(des_gc_noise_file):
    N_bpw = np.load(des_gc_noise_file)['cls']
    for i, N_bpwi in enumerate(N_bpw):
        cl_matrix[i, i] -= N_bpwi
else:
    des_N_mean_srad = des_N_mean / (4 * np.pi) * hp.nside2npix(des_nside)
    N_ell = des_mask.sum() / hp.nside2npix(des_nside) / des_N_mean_srad

    N_bpw = []
    ws = nmt.NmtWorkspace()
    fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(0, 0, 0, 0))
    ws.read_from(fname)
    for i, N_ell_mapi in enumerate(N_ell):
        N_bpw.append(ws.decouple_cell([N_ell_mapi * np.ones(3 * des_nside)])[0])
        cl_matrix[i, i] -= N_bpw[-1]

    N_bpw = np.array(N_bpw)

    np.savez(des_gc_noise_file,
             l=b.get_effective_ells(), cls=N_bpw)

# Compute DES shear noise

des_wl_noise_file = os.path.join(output_folder, "des_sh_metacal_rot0-10_noise_ns4096.npz")
if os.path.isfile(des_wl_noise_file):
    N_wl = np.load(des_wl_noise_file)['cls']
    for ibin, N_wli in enumerate(N_wl):
        index_bin = len(des_maps) + 2 * ibin
        cl_matrix[index_bin : index_bin + 2, index_bin : index_bin + 2] -= N_wli
else:
    N_wl = []
    for ibin in range(len(des_maps_we1)):
        rotated_cls = []
        for irot in range(10):
            map_file_e1 = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_rot{}_counts_e1_ns4096.fits'.format(ibin, irot))
            map_file_e2 = os.path.join(des_data_folder_gwl, 'map_metacal_bin{}_rot{}_counts_e2_ns4096.fits'.format(ibin, irot))

            map_we1 = hp.read_map(map_file_e1)
            map_we2 = hp.read_map(map_file_e2)

            map_e1 = (map_we1/des_mask_gwl[ibin] - (map_we1.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e2 = (map_we2/des_mask_gwl[ibin] - (map_we2.sum()/des_mask_gwl[ibin].sum())) / des_opm_mean[ibin]
            map_e1[np.isnan(maps_e1)] = 0.
            map_e2[np.isnan(maps_e2)] = 0.

            sq = map_e1[i]
            su = -map_e2[i]
            f = nmt.NmtField(des_mask_gwl[i], [sq, su])

            ws = nmt.NmtWorkspace()
            fname = os.path.join(output_folder, 'w22_{}{}.dat'.format(ibin, ibin))
            ws.read_from(fname)

            cls = ws.decouple_cell(nmt.compute_coupled_cell(f, f)).reshape((2, 2, -1))
            rotated_cls.append(cls)

        N_wl.append(np.mean(rotated_cls, axis=0))

        index_bin = len(des_maps) + 2 * ibin
        cl_matrix[index_bin : index_bin + 2, index_bin : index_bin + 2] -= N_wl[-1]

    np.savez(des_wl_noise_file,
             l=b.get_effective_ells(), cls=N_wl)

np.savez(os.path.join(output_folder, "cl_all_no_noise"),
         l=b.get_effective_ells(), cls=cl_matrix)

# ######################### Test ############################
# print(np.all(cl00_matrix[1,3] == cl00_matrix[3, 1]))
# print(np.all(cl00_matrix[2,3] == cl00_arr[10]))
# print(np.all(cl00_matrix[2,2] == cl00_arr[9]))
# ######################### Test ############################

if o.plot_stuff :
    plt.show()
