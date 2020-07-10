from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import sys
import common as co

# pylint: disable=C0103

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--nside', dest='nside', default=4096, type=int,
                  help='HEALPix nside param')
parser.add_option('--wltype', dest='wltype', default='metacal', type=str,
                  help='DES weak lensing shear measurement algorithm (metacal or im3shape)')
parser.add_option('--compute_cw', dest='compute_cw', default=False, action='store_true',
                  help='Compute covariance workspaces')
parser.add_option('--n_iter', dest='n_iter', default=3, type=int,
                  help='Change n_iter in NmtFields')
parser.add_option('--outdir', dest='outdir', default='', type=str,
                  help='Path to save output')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')

(o, args) = parser.parse_args()

##############################################################################
##############################################################################
##############################################################################
nside = o.nside
wltype = o.wltype

nrot = 20
gc_threshold = 0.5

data_folder = '/mnt/extraspace/damonge/S8z_data/derived_products'

# Output folder
if o.outdir:
    output_folder = o.outdir
elif nside == 4096:
    output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_new'.format(wltype)
else:
    output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_{}_{}_new'.format(wltype, nside)
os.makedirs(output_folder, exist_ok=True)

##############################################################################
############################## Set Binning ###################################
##############################################################################
# The ells_lim_bpw
b = co.get_NmtBin(nside)
fname = os.path.join(output_folder, 'l_bpw.txt')
np.savetxt(fname, b.get_effective_ells())
##############################################################################
############################ DES Clustering ##################################
##############################################################################

des_folder_gcl = 'des_clustering'
# des_nside = 4096
des_mask = 'mask_ns{}.fits'.format(nside)

des_data_folder = os.path.join(data_folder, des_folder_gcl)

des_mask_path = os.path.join(des_data_folder, des_mask)

# Read mask
# mask_lss = hp.ud_grade(hp.read_map(des_mask_path, verbose=False), nside_out=2048)
des_mask = hp.read_map(des_mask_path, verbose=False)
des_mask_good = des_mask > gc_threshold  # Can be generalized to accept a different threshold
des_mask[~des_mask_good] = 0
# Read maps (gg)
nmaps = 5
des_maps = []
for i in range(nmaps):
    map_file = os.path.join(des_data_folder, 'map_counts_w_bin{}_ns{}.fits'.format(i, nside))
    des_maps.append(hp.read_map(map_file))
des_maps = np.array(des_maps)

des_N_mean = des_maps[:, des_mask_good].sum(axis=1) / des_mask[des_mask_good].sum()
des_maps_dg = np.zeros(des_maps.shape)
des_maps_dg[:, des_mask_good] = des_maps[:, des_mask_good] / (des_N_mean[:, None] * des_mask[des_mask_good]) - 1

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
des_maps_e1 = []
des_maps_e2 = []
des_opm_mean = []
for i in range(4):
    fname = os.path.join(des_data_folder_gwl, 'sums_{}_bin{}.npz'.format(wltype, i))
    sums = np.load(fname)
    fname = os.path.join(des_data_folder_gwl, 'map_{}_bin{}_w_ns{}.fits'.format(wltype, i, nside))
    mask_gwl = hp.read_map(fname)
    fname = os.path.join(des_data_folder_gwl, 'map_{}_bin{}_we1_ns{}.fits'.format(wltype, i, nside))
    map_we1 = hp.read_map(fname)
    fname = os.path.join(des_data_folder_gwl, 'map_{}_bin{}_we2_ns{}.fits'.format(wltype, i, nside))
    map_we2 = hp.read_map(fname)

    mask_good = mask_gwl > 0.
    mask_gwl[~mask_good] = 0.
    opm_mean = sums['wopm'] / sums['w']

    map_e1 = np.zeros_like(map_we1)
    map_e2 = np.zeros_like(map_we2)
    map_e1[mask_good] = (map_we1[mask_good]/mask_gwl[mask_good]) / opm_mean
    map_e2[mask_good] = (map_we2[mask_good]/mask_gwl[mask_good]) / opm_mean

    des_mask_gwl.append(mask_gwl)
    des_maps_e1.append(map_e1)
    des_maps_e2.append(map_e2)
    des_opm_mean.append(opm_mean)

des_mask_gwl = np.array(des_mask_gwl)
des_maps_e1 = np.array(des_maps_e1)
des_maps_e2 = np.array(des_maps_e2)
des_opm_mean = np.array(des_opm_mean)

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

if nside in [2048, 4096]:
    fname = os.path.join(planck_data_folder, 'mask_ns{}.fits'.format(nside))
    planck_mask = hp.read_map(fname)
    fname = os.path.join(planck_data_folder, 'map_kappa_ns{}.fits'.format(nside))
    planck_map_kappa = hp.read_map(fname)
else:
    fname = os.path.join(planck_data_folder, 'mask_ns{}.fits'.format(4096))
    planck_mask = hp.ud_grade(hp.read_map(fname), nside_out=nside)
    fname = os.path.join(planck_data_folder, 'map_kappa_ns{}.fits'.format(4096))
    planck_map_kappa = hp.ud_grade(hp.read_map(fname), nside_out=nside)

##############################################################################
########################## Putting all together ##############################
##############################################################################

maps = np.empty((len(des_maps_dg) + len(des_maps_e1) * 2 + 1, planck_map_kappa.shape[0]))
maps[:len(des_maps_dg)] = des_maps_dg
maps[-1] = planck_map_kappa
nmaps = len(maps)

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


##############################################################################
############################# NaMaster stuff #################################
##############################################################################

##############################################################################
# Generate fields for each map
##############################################################################
fields = []
for mapi in des_maps_dg:
    fields.append(nmt.NmtField(des_mask, [mapi], n_iter=o.n_iter))

for i in range(des_maps_e1.shape[0]):
    sq = des_maps_e1[i]
    su = - des_maps_e2[i]
    f = nmt.NmtField(des_mask_gwl[i], [sq, su], n_iter=o.n_iter)
    fields += [f, f]

fields.append(nmt.NmtField(planck_mask, [planck_map_kappa], n_iter=o.n_iter))

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
# Generate covariance workspaces
##############################################################################
if o.compute_cw:
    cl_indices = []
    nmaps = len(maps)
    for i in range(nmaps):
        for j in range(i, nmaps):
            cl_indices.append([i, j])

    cov_indices = []
    for i, clij in enumerate(cl_indices):
        for j, clkl in enumerate(cl_indices[i:]):
            cov_indices.append(cl_indices[i] + cl_indices[i + j])

    for indices in cov_indices:
        i, j, k, l = indices
        mask1 = masks[i]
        mask2 = masks[j]
        mask3 = masks[k]
        mask4 = masks[l]
        fname = os.path.join(output_folder, 'cw{}{}{}{}.dat'.format(mask1, mask2, mask3, mask4))
        sys.stdout.write('cw{}{}{}{}.dat\n'.format(mask1, mask2, mask3, mask4))
        if not os.path.isfile(fname):
            cw = nmt.NmtCovarianceWorkspace()
            f1 = fields[i]
            f2 = fields[j]
            f3 = fields[k]
            f4 = fields[l]
            cw.compute_coupling_coefficients(f1, f2, f3, f4)
            cw.write_to(fname)

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

            cl_matrix[index1 : index1 + dof1, index2 : index2 + dof2] = cls * (-1)**(dof1 + dof2 + 2)  # To correct the minus sign in sh fields

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
des_gc_noise_file = os.path.join(output_folder, "des_w_cl_shot_noise_ns{}.npz".format(nside))
if os.path.isfile(des_gc_noise_file):
    N_bpw = np.load(des_gc_noise_file)['cls']
    for i, N_bpwi in enumerate(N_bpw):
        cl_matrix[i, i] -= N_bpwi
else:
    des_N_mean_srad = des_N_mean / (4 * np.pi) * hp.nside2npix(nside)
    N_ell = des_mask.sum() / hp.nside2npix(nside) / des_N_mean_srad

    N_bpw = []
    N_bpw_raw = []
    ws = nmt.NmtWorkspace()
    fname = os.path.join(output_folder, 'w{}{}_{}{}.dat'.format(0, 0, 0, 0))
    ws.read_from(fname)
    for i, N_ell_mapi in enumerate(N_ell):
        N_bpw.append(ws.decouple_cell([N_ell_mapi * np.ones(3 * nside)])[0])
        cl_matrix[i, i] -= N_bpw[-1]
        N_bpw_raw.append(N_ell_mapi * np.ones(3 * nside))

    N_bpw = np.array(N_bpw)
    N_bpw_raw = np.array(N_bpw_raw)
    noise_factor = np.mean(des_mask ** 2) * np.ones(N_ell.shape[0])

    np.savez(des_gc_noise_file,
             l=b.get_effective_ells(), cls=N_bpw, cls_raw=N_bpw_raw,
             noise_factor=noise_factor, cls_cov=N_bpw_raw/noise_factor[:, None])

# Compute DES shear noise

des_wl_noise_file = os.path.join(output_folder, "des_sh_{}_noise_ns{}.npz".format(wltype, nside))
if os.path.isfile(des_wl_noise_file):
    N_wl = np.load(des_wl_noise_file)['cls']
    for ibin, N_wli in enumerate(N_wl):
        index_bin = len(des_maps) + 2 * ibin
        cl_matrix[index_bin : index_bin + 2, index_bin : index_bin + 2] -= N_wli
else:
    N_wl = []
    N_wl_raw = []
    for ibin in range(len(des_maps_e1)):
        # rotated_cls = []
        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w22_{}{}.dat'.format(1 + ibin, 1 + ibin))
        ws.read_from(fname)

        # Analytical
        nlee = nlbb = co.get_shear_noise(ibin, wltype, nside, opm_mean=des_opm_mean[ibin])
        nleb = nlbe = 0 * nlee
        N_wl_raw.append(np.array([[nlee, nleb],[nlbe, nlbb]]))
        N_wl.append(ws.decouple_cell([nlee, nleb, nlbe, nlbb]).reshape((2, 2, -1)))

        index_bin = len(des_maps) + 2 * ibin
        cl_matrix[index_bin : index_bin + 2, index_bin : index_bin + 2] -= N_wl[-1]


    # Save noise from analytical exp
    noise_factor = np.mean(des_mask_gwl ** 2, axis=1)
    np.savez(des_wl_noise_file,
             l=b.get_effective_ells(), cls=N_wl, cls_raw=N_wl_raw, noise_factor=noise_factor,
             cls_cov = np.array(N_wl_raw)/noise_factor[:, None, None, None])


np.savez(os.path.join(output_folder, "cl_all_no_noise"),
         l=b.get_effective_ells(), cls=cl_matrix)

# Noise from rotated galaxies for crosscheck
fname_rots = os.path.join(output_folder, "des_sh_{}_rot0-{}_noise_ns{}.npz".format(wltype, nrot-1, nside))
frot = os.path.join(des_data_folder_gwl, "map_{}_bin1_rot1_we1_ns{}.fits".format(wltype, nside))
if not os.path.isfile(fname_rots) and os.path.isfile(frot):
    N_wl_rot = []
    for ibin in range(len(des_maps_e1)):
        ws = nmt.NmtWorkspace()
        fname = os.path.join(output_folder, 'w22_{}{}.dat'.format(1 + ibin, 1 + ibin))
        ws.read_from(fname)
        N_wl_rot.append(co.get_shear_noise_rot(ibin, wltype, nside, nrot=nrot,
                        mask=des_mask_gwl[ibin], opm_mean=des_opm_mean[ibin], ws=ws))
    np.savez(fname_rots,
             l=b.get_effective_ells(), cls=N_wl_rot)
#### End noise


# Split cls in files
bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
index_B = [6, 8, 10, 12]
co.split_cls_all_array(cl_matrix, b.get_effective_ells(), bins, index_B, output_folder)


# ######################### Test ############################
# print(np.all(cl00_matrix[1,3] == cl00_matrix[3, 1]))
# print(np.all(cl00_matrix[2,3] == cl00_arr[10]))
# print(np.all(cl00_matrix[2,2] == cl00_arr[9]))
# ######################### Test ############################

if o.plot_stuff :
    plt.show()
