from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
# parser.add_option('--nside', dest='nside', default=512, type=int,
#                   help='HEALPix nside param')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')

(o, args) = parser.parse_args()

output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/'

data_folder = '/mnt/extraspace/damonge/S8z_data/derived_products'
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

if o.plot_stuff:
    hp.mollview(des_mask)
    for mapi in des_maps_dg:
        hp.mollview(mapi)

#Set up binning scheme
fsky = np.mean(des_mask)
d_ell = int(1./fsky)
b = nmt.NmtBin(des_nside,nlb=d_ell)

#Generate an initial simulation
def get_fields(maps_dg):
    """
    Generate a simulated field.
    It returns two NmtField objects for a spin-0 and a spin-2 field.

    :param fsk: a fm.FlatMapInfo object.
    :param mask: a sky mask.
    :param w_cont: deproject any contaminants? (not implemented yet)
    """
    fields = []
    for mapi in maps_dg:
        fields.append(nmt.NmtField(des_mask, [mapi]))

    return fields

des_fields = get_fields(des_maps_dg)

#Compute mode-coupling matrix
#Use initial fields to generate coupling matrix
w00=nmt.NmtWorkspace();
fname = os.path.join(output_folder, 'des_w00_ns4096.dat')
if not os.path.isfile(fname): #spin0-spin0
    print("Computing 00")
    f0 = des_fields[0]  # All of them have same mask, so just need one w00
    w00.compute_coupling_matrix(f0, f0, b)
    w00.write_to(fname);
else :
    w00.read_from(fname)


# Compute cls
cl00_arr = []
for i, f0i in enumerate(des_fields):
    for f0j in des_fields[i:]:
        cl00 = w00.decouple_cell(nmt.compute_coupled_cell(f0i, f0j))[0]
        cl00_arr.append(cl00)

cl00_matrix = np.empty((len(des_fields), len(des_fields), len(cl00)),
                       dtype=cl00.dtype)
i, j = np.triu_indices(len(des_fields))
cl00_matrix[i, j] = cl00_arr
cl00_matrix[j, i] = cl00_arr

# Compute noise
des_N_mean_srad = des_N_mean / (4 * np.pi) * hp.nside2npix(des_nside)
N_ell = des_mask.sum() / hp.nside2npix(des_nside) / des_N_mean_srad

N_bpw = []
for i, N_ell_mapi in enumerate(N_ell):
    N_bpw.append(w00.decouple_cell([N_ell_mapi * np.ones(3 * des_nside)])[0])
    cl00_matrix[i, i] -= N_bpw[-1]

N_bpw = np.array(N_bpw)

np.savez(os.path.join(output_folder, "des_w_cl_ns4096"),
         l=b.get_effective_ells(), cls=cl00_matrix)
np.savez(os.path.join(output_folder, "des_w_cl_shot_noise_ns4096"),
         l=b.get_effective_ells(), cls=N_bpw)


# ######################### Test ############################
# print(np.all(cl00_matrix[1,3] == cl00_matrix[3, 1]))
# print(np.all(cl00_matrix[2,3] == cl00_arr[10]))
# print(np.all(cl00_matrix[2,2] == cl00_arr[9]))
# ######################### Test ############################

if o.plot_stuff :
    plt.show()
