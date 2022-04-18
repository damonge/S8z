#!/usr/bin/python
import numpy as np
import healpy as hp
import os

nside = 512

def to_fits(ibin):
    root = '/mnt/extraspace/damonge/S8z_data/outputs/'
    outdir = '/mnt/extraspace/gravityls_3/S8z/data/derived_products/des_shear/'
    goodpix = np.load(root + f'maps_metacal_bin{ibin}_ns{nside}_goodpix.npz')['pix']
    we = np.load(root + f'maps_metacal_bin{ibin}_ns{nside}_we.npz')
    w2s2 = np.load(root + f'maps_metacal_bin{ibin}_ns{nside}_w2s2.npz')['w2s2']
    w = np.load(root + f'maps_metacal_bin{ibin}_ns{nside}_w.npz')['w']
    # wpsfe = np.load(root + 'maps_metacal_bin{}_ns{nside}_wpsfe.npz'.format(ibin))

    zmap = np.zeros(hp.nside2npix(nside))
    fname = outdir + f'map_metacal_bin{ibin}_w_ns{nside}.fits'
    if not os.path.isfile(fname):
        zmap[goodpix] = w
        hp.write_map(fname, zmap)

    fname = outdir + f'map_metacal_bin{ibin}_we1_ns{nside}.fits'
    if not os.path.isfile(fname):
        zmap[goodpix] = we['e1']
        hp.write_map(fname, zmap)

    fname = outdir + f'map_metacal_bin{ibin}_we2_ns{nside}.fits'
    if not os.path.isfile(fname):
        zmap[goodpix] = we['e2']
        hp.write_map(fname, zmap)

    fname = outdir + f'sums_metacal_bin{ibin}.npz'
    if not os.path.isfile(fname):
        np.savez_compressed(fname, w2s2=np.sum(w2s2))

if __name__ == '__main__':
    for ibin in range(4):
        to_fits(ibin)
