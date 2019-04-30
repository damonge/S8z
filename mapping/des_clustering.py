import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
import os

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

# Read catalog
print("Reading catalog")
d = fits.open(predir_in + fname_cat)[1].data

# Create mask
print("Reading mask")
msk = hp.read_map(predir_in + fname_mask, verbose=False)
msk[msk == hp.UNSEEN] = 0
msk = hp.ud_grade(msk, nside_out=nside)
print("fsky = %lE, %lf" % (np.mean(msk), 4*np.pi*np.mean(msk)*(180/np.pi)**2))
hp.mollview(msk);
hp.write_map(predir_out + "mask_ns%d.fits" % nside, msk, overwrite=True)

# Create number count maps
def mk_nmap(cat, ns, mask, keep=None, do_weights=False):
    """
    ns -> nside
    cat -> recarray with fields 'DEC' and 'RA'
    keep -> mask to apply to the data (None for no mask)
    do_weights -> apply systematics weights
    return -> map of counts per pixel
    """
    # Apply cuts
    if keep is None:
        cat_use = cat
    else:
        cat_use = cat[keep]

    # Count galaxies per pixel
    npix = hp.nside2npix(ns)
    ipix = hp.ang2pix(ns,
                      np.radians(90 - cat_use['DEC']),
                      np.radians(cat_use['RA']))
    if do_weights:
        w = cat_use['weight']
    else:
        w = None
    nmap=np.bincount(ipix, weights=w, minlength=npix)+0.

    return nmap

# Create maps for each redshift bin
print("Per-bin count maps")
for iz, zr in enumerate(zbins):
    keep = ((d['ZREDMAGIC'] >= zr[0]) & (d['ZREDMAGIC'] < zr[1]))
    print(zr, np.sum(keep), np.sum(d[keep]['weight']))
    nmap = mk_nmap(d, nside, msk, keep=keep)
    hp.mollview(nmap)
    hp.write_map(predir_out + "map_counts_bin%d_ns%d.fits" % (iz, nside),
                 nmap, overwrite=True)
    nmap = mk_nmap(d, nside, msk, keep=keep, do_weights=True)
    hp.write_map(predir_out + "map_counts_w_bin%d_ns%d.fits" % (iz, nside),
                 nmap, overwrite=True)

# Create map from random catalog
print("Random count map")
dr = fits.open(predir_in + fname_ran)[1].data
nmap_r = mk_nmap(dr, nside, msk)
hp.write_map(predir_out + "map_counts_random_ns%d.fits" % nside,
             nmap_r, overwrite=True)

hp.mollview(nmap_r)
plt.show()
