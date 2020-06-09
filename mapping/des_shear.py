import numpy as np
import os
from astropy.io import fits
from maptools import get_fits_iterator, get_weighted_maps, get_weighted_sums
import sys
import healpy as hp
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--catalog", default='metacal', help="Catalog name")
parser.add_argument("--nside", default=512, type=int, help="Nside parameter")
parser.add_argument("--bin-number", default=0, type=int, help="Bin number")
parser.add_argument("--rotate", default=False, action="store_true",
                    help="Do random rotation?")
parser.add_argument("--seed", default=1234, type=int, help="Seed for random rotations")
parser.add_argument("--recompute", default=False, action="store_true",
                    help="Recompute even if files exist")
o = parser.parse_args()

use_im3shape = o.catalog == 'im3shape'
if o.rotate:
    np.random.seed(o.seed)


# Angular resolution
# Root data directory
predir_in = "/mnt/extraspace/damonge/S8z_data/"
if use_im3shape:
    # IM3shape shape catalog
    fname_cat = "DES_data/shear_catalog/y1a1-im3shape_v5_unblind_v2_matched_v4.fits"
    hdu_dndz = 2
    suffix_sums = 'im3shape'
else:
    # Metacal shape catalog
    fname_cat = "DES_data/shear_catalog/mcal-y1a1-combined-riz-unblind-v4-matched.fits"
    hdu_dndz = 1
    suffix_sums = 'metacal'
suffix_sums += "_bin%d" % o.bin_number
if o.rotate:
    suffix = suffix_sums + "_rot%d" % o.seed
else:
    suffix = suffix_sums
    
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
if not os.path.isfile(predir_out + "dndz_" + suffix_sums + ".txt"):
    print("Extracting N(z)")
    d = fits.open(predir_in + fname_bins)
    np.savetxt(predir_out + "dndz_" + suffix_sums + ".txt",
               np.transpose([d[hdu_dndz].data['Z_LOW'],
                             d[hdu_dndz].data['Z_MID'],
                             d[hdu_dndz].data['Z_HIGH'],
                             d[hdu_dndz].data['BIN%d' % (o.bin_number+1)]]))

# Iterators from files
print("Iterators")
# Joint iterators
def get_iterator_metacal(e1_mean, e2_mean):
    itr_cat = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'R11', 'R22',
                                 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_mcal'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_cat, itr_bin):
        ngal = len(m['ra'])
        if o.rotate:
            phi = 2 * np.pi * np.random.rand(ngal)
            c = np.cos(2*phi)
            s = np.sin(2*phi)
            e1 =  (m['e1']-e1_mean) * c + (m['e2']-e2_mean) * s
            e2 = -(m['e1']-e1_mean) * s + (m['e2']-e2_mean) * c
        else:
            e1 = m['e1']-e1_mean
            e2 = m['e2']-e2_mean

        dc={'ra':m['ra'],
            'dec':m['dec'],
            'e1':e1,
            'e2':e2,
            'sel':m['flags_select'],
            'w':np.ones(ngal),
            'bin':b['zbin_mcal']
        }
        yield dc

def get_iterator_metacal_sums():
    itr_cat = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'R11', 'R22',
                                 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_mcal'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_cat, itr_bin):
        ngal = len(m['ra'])

        e1 = m['e1']
        e2 = m['e2']
        dc={'ra':m['ra'],
            'dec':m['dec'],
            'e1':e1,
            'e2':e2,
            'opm':0.5*(m['R11']+m['R22']),
            'sel':m['flags_select'],
            'w':np.ones(ngal),
            'ws2': 0.5*(e1**2+e2**2),
            'we1': e1,
            'we2': e2,
            'bin':b['zbin_mcal']
        }
        yield dc

def get_iterator_im3shap(e1_mean, e2_mean):
    itr_im3 = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'm', 'c1', 'c2',
                                 'weight', 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_im3'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_im3, itr_bin):
        ngal = len(m['ra'])
        if o.rotate:
            phi = 2 * np.pi * np.random.rand(ngal)
            c = np.cos(2*phi)
            s = np.sin(2*phi)
            e1 =  (m['e1']-m['c1']-e1_mean) * c + (m['e2']-m['c2']-e2_mean) * s
            e2 = -(m['e1']-m['c1']-e1_mean) * s + (m['e2']-m['c2']-e2_mean) * c
        else:
            e1 = m['e1']-m['c1']-e1_mean
            e2 = m['e2']-m['c2']-e2_mean

        dc={'ra':m['ra'],
            'dec':m['dec'],
            'e1':e1,
            'e2':e2,
            'sel':m['flags_select'],
            'w':m['weight'],
            'bin':b['zbin_im3']
        }
        yield dc

def get_iterator_im3shap_sums():
    itr_im3 = get_fits_iterator(predir_in + fname_cat,
                                ['coadd_objects_id', 'e1', 'e2', 'm', 'c1', 'c2',
                                 'weight', 'ra', 'dec', 'flags_select'],
                                nrows_per_chunk=1000000)
    itr_bin = get_fits_iterator(predir_in + fname_binning,
                                ['coadd_objects_id', 'zbin_im3'],
                                nrows_per_chunk=1000000)
    for m,b in zip(itr_im3, itr_bin):
        e1 = m['e1']-m['c1']
        e2 = m['e2']-m['c2']

        dc={'ra':m['ra'],
            'dec':m['dec'],
            'e1':e1,
            'e2':e2,
            'opm':1+m['m'],
            'sel':m['flags_select'],
            'w':m['weight'],
            'ws2':m['weight']*0.5*(e1**2+e2**2),
            'we1':m['weight']*e1,
            'we2':m['weight']*e2,
            'bin':b['zbin_im3']
        }
        yield dc

# Catalog masks for each bin
masks = [[['tag','bin',int(o.bin_number)],
          ['tag','sel',0],
          ['range','dec',-90.,-35.]]]

# Maps
print("Sums")
if o.recompute or (not os.path.isfile(predir_out + "sums_" + suffix_sums + ".npz")):
    if use_im3shape:
        itr = get_iterator_im3shap_sums()
    else:
        itr = get_iterator_metacal_sums()
    n, w, fl = get_weighted_sums(itr, name_weight='w',
                                 names_field=['e1', 'e2', 'ws2',
                                              'we1', 'we2', 'w', 'opm'],
                                 masks=masks)
    sums = {'count': n,
            'w': w,
            'we1': fl[0],
            'we2': fl[1],
            'w2s2': fl[2],
            'w2e1': fl[3],
            'w2e2': fl[4],
            'w2': fl[5],
            'wopm': fl[6]}
    e1_mean = sums['we1'] / sums['w']
    e2_mean = sums['we2'] / sums['w']
    sums['we1'] = sums['we1'] - e1_mean * sums['w']
    sums['we2'] = sums['we2'] - e2_mean * sums['w']
    sums['w2s2'] = (sums['w2s2'] - e1_mean * sums['w2e1'] -
                    e2_mean * sums['w2e2'] +
                    0.5 * (e1_mean**2+e2_mean**2) * sums['w2'])
    sums['w2e1'] = sums['w2e1'] - e1_mean * sums['w2']
    sums['w2e2'] = sums['w2e2'] - e2_mean * sums['w2']
    sums['e1_mean'] = e1_mean
    sums['e2_mean'] = e2_mean
    np.savez(predir_out + "sums_" + suffix_sums + ".npz",
             **sums)
else:
    sums = dict(np.load(predir_out + "sums_" + suffix_sums + ".npz").items())
e1_mean = sums['e1_mean']
e2_mean = sums['e2_mean']

print("Maps")
if use_im3shape:
    itr = get_iterator_im3shap(e1_mean, e2_mean)
else:
    itr = get_iterator_metacal(e1_mean, e2_mean)

nm, wm, fl = get_weighted_maps(itr, o.nside, 'ra', 'dec',
                               name_weight='w',
                               names_field=['e1', 'e2'],
                               masks=masks)

# Write output maps
print("Writing outputs")
hp.write_map(predir_out + "map_" + suffix + "_we1_ns%d.fits" % o.nside,
             fl[0], overwrite=True)
hp.write_map(predir_out + "map_" + suffix + "_we2_ns%d.fits" % o.nside,
             fl[1], overwrite=True)
if not o.rotate:
    hp.write_map(predir_out + "map_" + suffix + "_counts_ns%d.fits" % o.nside,
                 nm, overwrite=True)
    hp.write_map(predir_out + "map_" + suffix + "_w_ns%d.fits" % o.nside,
                 wm, overwrite=True)
