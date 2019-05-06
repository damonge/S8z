import numpy as np
import healpy as hp
from astropy.io import fits
import os
from maptools import get_fits_iterator, get_weighted_maps

# Angular resolution
nside = 4096
# Root data directory
predir_in = "/mnt/bluewhale/damonge/S8z_data/"
# Catalog file
fname_cat = "DES_data/redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits"
# Mask file
fname_mask = "DES_data/redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits"
# Random catalog
fname_ran = "DES_data/redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_RANDOMS.fits"
# Final data vector (for N(z)s)
fname_dvc = "DES_data/data_vector/2pt_NG_mcal_1110.fits"
# Photo-z bin edges
zbins = [[0.15, 0.30],
         [0.30, 0.45],
         [0.45, 0.60],
         [0.60, 0.75],
         [0.75, 0.90]]
nbins = len(zbins)
# Output prefix
predir_out = "/mnt/bluewhale/damonge/S8z_data/derived_products/des_clustering/"
os.system("mkdir -p " + predir_out)

# Extract redshift distributions
df = fits.open(predir_in + fname_dvc)[7].data
for i in range(5):
    np.savetxt(predir_out + "dndz_bin%d.txt" % i,
               np.transpose([df['Z_LOW'],df['Z_MID'],df['Z_HIGH'],df['BIN%d' % (i+1)]]),
               header='[0]-z_ini [1]-z_mid [2]-z_end [3]-dndz')

# Create mask
print("Reading mask")
msk = hp.read_map(predir_in + fname_mask, verbose=False)
msk[msk == hp.UNSEEN] = 0
msk = hp.ud_grade(msk, nside_out=nside)
print(" fsky = %lE, %lf" % (np.mean(msk),
                            4*np.pi*np.mean(msk)*(180/np.pi)**2))
print(" Writing to file")
hp.write_map(predir_out + "mask_ns%d.fits" % nside, msk, overwrite=True)

# Create maps for each redshift bin
print("Creating maps")
# Iterator that reads fits file in chunks
itr = get_fits_iterator(predir_in + fname_cat, ['RA', 'DEC', 'weight', 'ZREDMAGIC'],
                        nrows_per_chunk=1000000)
# Generate masks for each redshift bin
masks = []
for iz, zr in enumerate(zbins):
    masks.append(['range','ZREDMAGIC',zr[0],zr[1]])
# One final map with all galaxies in it
masks.append(['all'])
# Use iterator and masks to create maps
nmaps, wmaps, _ = get_weighted_maps(itr, nside, 'RA', 'DEC',
                                    name_weight='weight', masks=masks)
print(" Writing to file")
# Write to file
for iz in range(len(zbins)):
    print(iz)
    hp.write_map(predir_out + "map_counts_bin%d_ns%d.fits" % (iz, nside),
                 nmaps[iz], overwrite=True)
    hp.write_map(predir_out + "map_counts_w_bin%d_ns%d.fits" % (iz, nside),
                 wmaps[iz], overwrite=True)
hp.write_map(predir_out + "map_counts_all_ns%d.fits" % nside,
             nmaps[-1], overwrite=True)
hp.write_map(predir_out + "map_counts_w_all_ns%d.fits" % nside,
             wmaps[-1], overwrite=True)
    
# Create map from random catalog
print("Random count map")
itr = get_fits_iterator(predir_in + fname_ran, ['RA', 'DEC'],
                        nrows_per_chunk=1000000)
nmap_r, _, _ = get_weighted_maps(itr, nside, "RA", "DEC")
print(nmap_r.shape)
print(" Writing to file")
hp.write_map(predir_out + "map_counts_random_ns%d.fits" % nside,
             nmap_r, overwrite=True)
