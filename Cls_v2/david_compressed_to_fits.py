#!/usr/bin/python
import numpy as np
import healpy as hp
import os

def to_fits(ibin):
    root = '/mnt/extraspace/damonge/S8z_data/outputs/'
    outdir = '/mnt/extraspace/gravityls_3/S8z/data/derived_products/des_shear/'
    goodpix = np.load(root + 'maps_metacal_bin{}_ns1024_goodpix.npz'.format(ibin))['pix']
    we = np.load(root + 'maps_metacal_bin{}_ns1024_we.npz'.format(ibin))
    w2s2 = np.load(root + 'maps_metacal_bin{}_ns1024_w2s2.npz'.format(ibin))['w2s2']
    w = np.load(root + 'maps_metacal_bin{}_ns1024_w.npz'.format(ibin))['w']
    # wpsfe = np.load(root + 'maps_metacal_bin{}_ns1024_wpsfe.npz'.format(ibin))

    zmap = np.zeros(hp.nside2npix(1024))
    fname = outdir + 'map_metacal_bin{}_w_ns1024.fits'.format(ibin)
    if not os.path.isfile(fname):
        zmap[goodpix] = w
        hp.write_map(fname, zmap)

    fname = outdir + 'map_metacal_bin{}_we1_ns1024.fits'.format(ibin)
    if not os.path.isfile(fname):
        zmap[goodpix] = we['e1']
        hp.write_map(fname, zmap)

    fname = outdir + 'map_metacal_bin{}_we2_ns1024.fits'.format(ibin)
    if not os.path.isfile(fname):
        zmap[goodpix] = we['e2']
        hp.write_map(fname, zmap)

    fname = outdir + 'sums_metacal_bin{}.npz'.format(ibin)
    if not os.path.isfile(fname):
        np.savez_compressed(fname, w2s2=np.sum(w2s2))

if __name__ == '__main__':
    for ibin in range(4):
        to_fits(ibin)
