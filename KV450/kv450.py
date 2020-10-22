#!/usr/bin/python
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pandas as pd

class KV450():
    def __init__(self, catalog_list, nside=4096, band9=True):
        self.catalog_list = catalog_list
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.band9 = band9
        self.column_names = ['SG_FLAG', 'GAAP_Flag_ugriZYJHKs',
                             'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             'e1_correction', 'e2_correction',
                             'weight']
        self.zbin_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
        # tomo_data has <c> removed per tile and tomo
        self.tomo_data = self._load_tomo_data()

    def get_psf_map(self, zbin):
        pass
    def get_shear_map(self, zbin):
        pass
    def get_star_map(self, zbin):
        pass
    def get_mask(self, zbin):
        pass

    def _load_tomo_data(self):
        zedges = self.zbin_edges
        data = [Table() for i in range(zedges.size - 1)]

        for catalog in self.catalog_list:
            data_cat = self._load_data_from_catalog(catalog)
            for i in range(zedges.size - 1):
                sel = (data_cat['Z_B'] > zedges[i]) * (data_cat['Z_B'] <= zedges[i + 1])
                data[i] = vstack([data[i], data_cat[sel]])

        for d in data:
            self._remove_additive_bias(d)
            print(len(d))

        return data

    def _load_data_from_catalog(self, catalog):
        print('Loading catalog:', catalog)
        data = pd.DataFrame()
        data = Table.read(catalog, format='fits')
        data = data[self.column_names]

        if self.band9:
            # Not optimal
            mask = data['GAAP_Flag_ugriZYJHKs'] == 0
            data = data[mask]

        # DataFrame modified directly
        self._remove_additive_bias(data)

        return data

    def _remove_additive_bias(self, data):
        print(np.mean(data['e1_correction']))
        print(np.mean(data['e2_correction']))
        data['e1_correction'] -= np.mean(data['e1_correction'])
        data['e2_correction'] -= np.mean(data['e2_correction'])

    def _get_ipix_for_data(self, data):
        phi = np.radians(data['RAJ2000'])
        theta = np.radians(90 - data['DECJ2000'])

        ipix = hp.ang2pix(self.nside, theta, phi)
        return ipix

    def _compute_shear_map(self, zbin):
        pass

    def _compute_psf_map(self, zbin):
        pass

    def _compute_star_map(self, zbin):
        pass

    def _compute_mask(self, zbin):
        data = self.tomo_data[zbin]
        ipix = self._get_ipix_for_data(data)


if __name__ == "__main__":
    from glob import glob
    lsfiles = glob('/mnt/extraspace/damonge/S8z_data/KiDS_data/shear_KV450_catalog/*')
    kv450 = KV450(lsfiles)





