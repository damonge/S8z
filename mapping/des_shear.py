import numpy as np
import os
from astropy.io import fits
from maptools import get_fits_iterator, get_weighted_maps
import sys
import healpy as hp

if len(sys.argv)<4:
    print("Usage: des_shear.py catalog nside bin_number [do_rotate, seed]")
    exit(1)

use_im3shape = sys.argv[1]=='im3shape'
nside = int(sys.argv[2])
bin_no = int(sys.argv[3])
if len(sys.argv)>4:
    random_rotate = sys.argv[4]=='do_rotate'
    seed = int(sys.argv[5])
    np.random.seed(seed)
else:
    random_rotate = False

# Angular resolution
# Root data directory
predir_in = "/mnt/extraspace/damonge/S8z_data/"
if use_im3shape:
    # IM3shape shape catalog
    fname_cat = "DES_data/shear_catalog/y1a1-im3shape_v5_unblind_v2_matched_v4.fits"
    hdu_dndz = 2
    suffix = 'im3shape'
else:
    # Metacal shape catalog
    fname_cat = "DES_data/shear_catalog/mcal-y1a1-combined-riz-unblind-v4-matched.fits"
    hdu_dndz = 1
    suffix = 'metacal'
suffix += "_bin%d" % bin_no
if random_rotate:
    suffix+="_rot%d" % seed
    
# Binning file
fname_binning = "DES_data/shear_catalog/y1_source_redshift_binning_v1.fits"
# N(z) file
fname_bins = "DES_data/shear_catalog/y1_redshift_distributions_v1.fits"
# Output prefix
predir_out = "/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/"
os.system("mkdir -p " + predir_out)
# Redshift bins
nbins = 4

# Extract redshift distributions
print("Extracting N(z)")
if not random_rotate:
    d = fits.open(predir_in + fname_bins)
    np.savetxt(predir_out + "dndz_" + suffix + ".txt",
               np.transpose([d[hdu_dndz].data['Z_LOW'],
                             d[hdu_dndz].data['Z_MID'],
                             d[hdu_dndz].data['Z_HIGH'],
                             d[hdu_dndz].data['BIN%d' % (bin_no+1)]]))

# Iterators from files
print("Iterators")
# Joint iterators
def get_iterator_metacal():
    itr_cat = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'R11', 'R22',
                                 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_mcal'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_cat, itr_bin):
        ngal = len(m['ra'])
        if random_rotate:
            phi = 2 * np.pi * np.random.rand(ngal)
            c = np.cos(2*phi)
            s = np.sin(2*phi)
            e1 =  m['e1'] * c + m['e2'] * s
            e2 = -m['e1'] * s + m['e2'] * c
        else:
            e1 = m['e1']
            e2 = m['e2']

        dc={'ra':m['ra'],
            'dec':m['dec'],
            'f1':e1,
            'f2':e2,
            'opm':0.5*(m['R11']+m['R22']),
            'sel':m['flags_select'],
            'w':np.ones(ngal),
            'bin':b['zbin_mcal']
        }
        yield dc

def get_iterator_im3shap():
    itr_im3 = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'm', 'c1', 'c2',
                                 'weight', 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_im3'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_im3, itr_bin):
        ngal = len(m['ra'])
        if random_rotate:
            phi = 2 * np.pi * np.random.rand(ngal)
            c = np.cos(2*phi)
            s = np.sin(2*phi)
            e1 =  (m['e1']-m['c1']) * c + (m['e2']-m['c2']) * s
            e2 = -(m['e1']-m['c1']) * s + (m['e2']-m['c2']) * c
        else:
            e1 = m['e1']-m['c1']
            e2 = m['e2']-m['c2']

        dc={'ra':m['ra'],
            'dec':m['dec'],
            'f1':e1,
            'f2':e2,
            'opm':1+m['m'],
            'sel':m['flags_select'],
            'w':m['weight'],
            'bin':b['zbin_im3']
        }
        yield dc

if use_im3shape:
    itr = get_iterator_im3shap()
else:
    itr = get_iterator_metacal()

# Catalog masks for each bin
masks = [[['tag','bin',int(bin_no)],
          ['tag','sel',0],
          ['range','dec',-90.,-35.]]]

# Maps
print("Metacal maps")
nm, wm, fl = get_weighted_maps(itr, nside, 'ra', 'dec',
                               name_weight='w',
                               names_field=['opm','f1','f2'],
                               masks=masks)
print(nm.shape, wm.shape, fl.shape)

# Write output maps
print("Writing outputs")
if not random_rotate:
    hp.write_map(predir_out + "map_" + suffix + "_counts_ns%d.fits" % nside,
                 nm, overwrite=True)
    hp.write_map(predir_out + "map_" + suffix + "_counts_w_ns%d.fits" % nside,
                 wm, overwrite=True)
    hp.write_map(predir_out + "map_" + suffix + "_counts_opm_ns%d.fits" % nside,
                 fl[0], overwrite=True)
hp.write_map(predir_out + "map_" + suffix + "_counts_e1_ns%d.fits" % nside,
             fl[1], overwrite=True)
hp.write_map(predir_out + "map_" + suffix + "_counts_e2_ns%d.fits" % nside,
             fl[2], overwrite=True)
