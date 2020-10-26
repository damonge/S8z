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
        return self._compute_psf_map(zbin)

    def get_shear_map(self, zbin):
        return self._compute_shear_map(zbin)

    def get_star_map(self, zbin):
        return self._compute_star_map(zbin)

    def get_star_mask(self, zbin):
        return self._compute_star_mask(zbin)

    def get_mask(self, zbin):
        return self._compute_mask(zbin)

    def _get_galaxy_data(self, zbin):
        data = self.tomo_data[zbin]
        sel = data['SG_FLAG'] == 1
        return data[sel]

    def _get_star_data(self, zbin):
        data = self.tomo_data[zbin]
        sel = data['SG_FLAG'] == 0
        return data[sel]

    def _load_tomo_data(self):
        zedges = self.zbin_edges
        data = [Table() for i in range(zedges.size - 1)]

        for catalog in self.catalog_list:
            data_cat = self._load_data_from_catalog(catalog)
            for i in range(zedges.size - 1):
                sel = (data_cat['Z_B'] > zedges[i]) * (data_cat['Z_B'] <= zedges[i + 1])
                data[i] = vstack([data[i], data_cat[sel]])

        for i, d in enumerate(data):
            self._remove_additive_bias(d)
            self._add_ipix_to_data(d)
            print('Tomographic bin {} has {} elements'.format(i, len(d)))

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
        sel_gals = data['SG_FLAG'] == 1
        print(np.mean(data[sel_gals]['e1_correction']))
        print(np.mean(data[sel_gals]['e2_correction']))
        data[sel_gals]['e1_correction'] -= np.mean(data[sel_gals]['e1_correction'])
        data[sel_gals]['e2_correction'] -= np.mean(data[sel_gals]['e2_correction'])

    def _add_ipix_to_data(self, data):
        phi = np.radians(data['ALPHA_J2000'])
        theta = np.radians(90 - data['DELTA_J2000'])


        ipix = hp.ang2pix(self.nside, theta, phi)
        data['ipix'] = ipix

    def _compute_shear_map(self, zbin):
        data = self._get_galaxy_data(zbin)
        we1 = np.bincount(data['ipix'], weights=data['weight']*data['e1_correction'], minlength=self.npix)
        we2 = np.bincount(data['ipix'], weights=data['weight']*data['e2_correction'], minlength=self.npix)
        w2s2 = np.bincount(data['ipix'], weights=data['weight']**2 * (data['e2_correction']**2 + data['e2_correction']**2), minlength=self.npix)

        return we1, we2, w2s2

    def _compute_psf_map(self, zbin):
        data = self._get_galaxy_data(zbin)
        we1 = np.bincount(data['ipix'], weights=data['weight']*data['PSF_e1'], minlength=self.npix)
        we2 = np.bincount(data['ipix'], weights=data['weight']*data['PSF_e2'], minlength=self.npix)
        w2s2 = np.bincount(data['ipix'], weights=data['weight']**2 * (data['PSF_e1']**2 + data['PSF_e2']**2), minlength=self.npix)

        return we1, we2, w2s2

    def _compute_star_map(self, zbin):
        data = self._get_star_data(zbin)
        counts = np.bincount(data['ipix'], minlength=self.npix)
        return counts

    def _compute_star_mask(self, zbin, w2=True):
        data = self._get_star_data(zbin)
        mask = np.bincount(data['ipix'], weights=data['weight'], minlength=self.npix)
        if not w2:
            return mask
        w2 = np.bincount(data['ipix'], weights=data['weight']**2, minlength=self.npix)
        return mask, w2

    def _compute_mask(self, zbin):
        data = self._get_galaxy_data(zbin)
        mask = np.bincount(data['ipix'], weights=data['weight'], minlength=self.npix)
        return mask



if __name__ == "__main__":
    from glob import glob
    from matplotlib import pyplot as plt
    lsfiles = glob('/mnt/extraspace/damonge/S8z_data/KiDS_data/shear_KV450_catalog/*')
    kv450 = KV450(lsfiles)
    we1, we2, _ = kv450.get_shear_map(0)
    w = kv450.get_mask(0)
    hp.mollview(we1 / w, title='KV450 - 1st zbin - e1', min=-0.1, max=0.1)
    plt.savefig('kv450_e1.png')
    hp.mollview(we2 / w, title='KV450 - 1st zbin - e2', min=-0.1, max=0.1)
    plt.savefig('kv450_e2.png')





