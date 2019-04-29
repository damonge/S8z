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

# Read catalog
print("Reading catalog")
d = fits.open(predir_in + fname_cat)[1].data

# Create mask
print("Reading mask")
msk = hp.read_map(predir_in + fname_mask, verbose=False)
msk[msk == hp.UNSEEN] = 0
msk = hp.ud_grade(msk, nside_out=nside)
print("fsky = %lE, %lf" % (np.mean(msk), 4*np.pi*np.mean(msk)*(180/np.pi)**2))
hp.write_map(predir_out + "mask_ns%d.fits" % nside, msk, overwrite=True)

# Create number count maps
def mk_nmap(cat, ns, mask, keep=None):
    """
    ns -> nside
    cat -> recarray with fields 'DEC' and 'RA'
    keep -> mask to apply to the data (None for no mask)
    return -> map of counts per pixel
    """
    # Apply cuts
    if keep is None:
        cat_use = cat
    else:
        cat_use = cat[keep]

    # Count galaxies per pixel
    npix=hp.nside2npix(ns)
    ipix=hp.ang2pix(ns,
                    np.radians(90 - cat_use['DEC']),
                    np.radians(cat_use['RA']))
    nmap=np.bincount(ipix, minlength=npix)+0.

    return nmap

# Create maps for each redshift bin
print("Per-bin count maps")
nmaps = np.zeros([nbins, hp.nside2npix(nside)])
for iz, zr in enumerate(zbins):
    keep = ((d['ZREDMAGIC'] >= zr[0]) & (d['ZREDMAGIC'] < zr[1]))
    print(zr, np.sum(keep))
    nmaps[iz] = mk_nmap(d, nside, msk, keep=keep)
    hp.write_map(predir_out + "map_counts_bin%d_ns%d.fits" % (iz, nside),
                 nmaps[iz], overwrite=True)

# Create map from random catalog
print("Random count map")
dr = fits.open(predir_in + fname_ran)[1].data
nmap_r = mk_nmap(dr, nside, msk)
hp.write_map(predir_out + "map_counts_random_ns%d.fits" % nside,
             nmap_r, overwrite=True)

print("Plotting")
hp.mollview(msk);
hp.mollview(nmap_r)
for m in nmaps:
    hp.mollview(m)
plt.show()
