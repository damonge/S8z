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
parser.add_option('--prefix-out',dest='prefix_out',default='run',type=str,
                  help='Output prefix')
parser.add_option('--nside', dest='nside', default=512, type=int,
                  help='HEALPix nside param')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants (ignore for now)')
parser.add_option('--nls-contaminants', dest='nls_cont', default=0, type=int,
                  help='Number of Large Scales contaminants')
parser.add_option('--nss-contaminants', dest='nss_cont', default=0, type=int,
                  help='Number of Small Scales contaminants')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--no-deproject',dest='no_deproject',default=False,action='store_true',
                  help='Set if you will include contaminants but won\'t clean them (ignore for now)')
parser.add_option('--no-debias',dest='no_debias',default=False,action='store_true',
                  help='Set if you will include contaminants, clean them but won\'t correct for the bias (ignore for now)')
parser.add_option('--low-noise-ee-bb',dest='low_noise_ee_bb',default=False,action='store_true',
                  help='Set if you want the noise for ee and bb modes be multiplied by 1e-2')

(o, args) = parser.parse_args()

# nsims=o.isim_end-o.isim_ini+1

data_folder = '/mnt/bluewhale/damonge/S8z_data/derived_products'
des_folder_gcl = 'des_clustering'
des_mask = 'mask_ns4096.fits'

des_data_folder = os.path.join(data_folder, des_folder_gcl)

des_mask_path = os.path.join(des_data_folder, des_mask)

# Read mask
# mask_lss = hp.ud_grade(hp.read_map(des_mask_path, verbose=False), nside_out=2048)
des_mask = hp.read_map(des_mask_path, verbose=False)
# Read maps (gg)
nmaps = 5
des_maps = np.zeros((nmaps, 4096))
for i in range(nmaps):
    map_file = os.path.join(des_data_folder, 'maps_counts_w_bin{}_ns4096.fits'.format(i))
    des_maps[i] = hp.read_map(map_file)

N_mean = des_maps.sum(axis=1) / des_mask.sum()
des_maps_dg = des_maps / (N_mean * des_mask) - 1
des_maps_dg[np.isnan(des_maps_dg)] = 0.


###### Test ######

for i, mapi in enumerate(des_maps_dg):
    check = des_maps[i] / (N_mean * des_mask) - 1
    check[np.isnan(check)] = 0

    print(np.all(mapi == check))

###### Test ######

sys.exit()

if o.plot_stuff:
    hp.mollview(des_mask)
    for mapi in des_maps_dg:
        hp.mollview(mapi)

#Set up binning scheme
fsky = np.mean(des_mask)
d_ell = int(1./fsky)
b = nmt.NmtBin(o.nside,nlb=d_ell)

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
if not os.path.isfile(o.prefix_out+"_w00.dat"): #spin0-spin0
    print("Computing 00")
    f0 = des_fields[0]  # All of them have same mask, so just need one w00
    w00.compute_coupling_matrix(f0, f0, b)
    w00.write_to(o.prefix_out+"_w00.dat");
else :
    w00.read_from(o.prefix_out+"_w00.dat")


#Compute mean and variance over nsims simulations
cl00_arr = []
for i, f0i in enumerate(des_fields):
    for f0j in des_fields[i:]:
        cl00 = w00.decouple_cell(nmt.compute_coupled_cell(f0i, f0j))
        cl00_arr.append(cl00)

cl00_matrix = np.empty((len(des_fields), len(des_fields), len(cl00)),
                       dtype=cl00.dtype)
i, j = np.triu_indices(len(des_fields))
cl00_matrix[i, j] = cl00_arr
cl00_matrix[j, i] = cl00_arr


######################### Test ############################
print(np.all(cl00_matrix[1,3] == cl00_matrix[3, 1]))
print(np.all(cl00_matrix[2,3] == cl00_arr[10]))
print(np.all(cl00_matrix[2,2] == cl00_arr[9]))

np.savez(os.path.join(des_data_folder, "cl_ns4096"),
         l=b.get_effective_ells(), cls=cl00_matrix)

if o.plot_stuff :
    plt.show()