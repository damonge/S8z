#!/usr/bin/python
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pandas as pd

class KV450():
    def __init__(self, catalog_list, nside=4096, band9=True, dndz_list=None):
        self.catalog_list = catalog_list
        self.dndz_list = dndz_list
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.band9 = band9
        self.column_names = ['SG_FLAG', 'GAAP_Flag_ugriZYJHKs',
                             'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             'bias_corrected_e1', 'bias_corrected_e2',
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

    def get_nzdz(self, zbin):
        if self.dndz_list is None:
            raise ValueError('If you want to get_nzdz you need to provide dndz_list')
        return self._read_nzdz(zbin)

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
                data_zbin = data_cat[sel]
                self._remove_additive_bias(data_zbin)
                data[i] = vstack([data[i], data_zbin])

        for i, d in enumerate(data):
            self._remove_multiplicative_bias(d, i)
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
        # self._remove_additive_bias(data)

        return data

    def _remove_additive_bias(self, data):
        sel_gals = data['SG_FLAG'] == 1
        data['bias_corrected_e1'][sel_gals] -= np.mean(data['bias_corrected_e1'][sel_gals])
        data['bias_corrected_e2'][sel_gals] -= np.mean(data['bias_corrected_e2'][sel_gals])

    def _remove_multiplicative_bias(self, data, zbin):
        # Values from Table 2 of 1812.06076 (KV450 cosmo paper)
        m = (-0.017, -0.008, -0.015, 0.010, 0.006)
        sel_gals = data['SG_FLAG'] == 1
        data['bias_corrected_e1'][sel_gals] /= 1 + m[zbin]
        data['bias_corrected_e2'][sel_gals] /= 1 + m[zbin]


    def _add_ipix_to_data(self, data):
        phi = np.radians(data['ALPHA_J2000'])
        theta = np.radians(90 - data['DELTA_J2000'])

        ipix = hp.ang2pix(self.nside, theta, phi)
        data['ipix'] = ipix

    def _compute_shear_map(self, zbin):
        data = self._get_galaxy_data(zbin)
        we1 = np.bincount(data['ipix'], weights=data['weight']*data['bias_corrected_e1'], minlength=self.npix)
        we2 = np.bincount(data['ipix'], weights=data['weight']*data['bias_corrected_e2'], minlength=self.npix)
        w2s2 = np.bincount(data['ipix'], weights=data['weight']**2 * 0.5 * (data['bias_corrected_e1']**2 + data['bias_corrected_e2']**2), minlength=self.npix)

        return we1, we2, w2s2

    def _compute_psf_map(self, zbin):
        data = self._get_galaxy_data(zbin)
        we1 = np.bincount(data['ipix'], weights=data['weight']*data['PSF_e1'], minlength=self.npix)
        we2 = np.bincount(data['ipix'], weights=data['weight']*data['PSF_e2'], minlength=self.npix)
        w2s2 = np.bincount(data['ipix'], weights=data['weight']**2 * 0.5 * (data['PSF_e1']**2 + data['PSF_e2']**2), minlength=self.npix)

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

    def _read_nzdz(self, zbin):
        return np.loadtxt(self.dndz_list[zbin], unpack=True)





if __name__ == "__main__":
    from glob import glob
    from matplotlib import pyplot as plt
    lsfiles = glob('/mnt/extraspace/damonge/S8z_data/KiDS_data/shear_KV450_catalog/*')
    kv450 = KV450(lsfiles)
    for i in range(5):
        we1, we2, w2s2 = kv450.get_shear_map(i)
        w = kv450.get_mask(i)
        hp.write_map('kv450_we1_bin{}.fits'.format(i), we1)
        hp.write_map('kv450_we2_bin{}.fits'.format(i), we2)
        hp.write_map('kv450_w2s2_bin{}.fits'.format(i), w2s2)
        hp.write_map('kv450_w_bin{}.fits'.format(i), w)
        np.savez_compressed('kv450_sums_bin{}.npz'.format(i), w2s2=np.sum(w2s2))
        # hp.mollview(we1 / w, title='KV450 - 1st zbin - e1', min=-0.5, max=0.5)
        # plt.savefig('kv450_e1_bin{}.png'.format(i))
        # hp.mollview(we2 / w, title='KV450 - 1st zbin - e2', min=-0.5, max=0.5)
        # plt.savefig('kv450_e2_bin{}.png'.format(i))





