#!/usr/bin/python

import sys
sys.path.append('../../DEScls/')
import xcell
import healpy as hp
import pyccl as ccl
from matplotlib import pyplot as plt


def scale_bin_map(mask, c):
    goodpix = mask != 0
    mask[goodpix] = mask[goodpix] / mask[goodpix] * c
    return mask



if __name__ == '__main__':
    nside = 512

    DESgc__0 = xcell.mappers.MapperDESY1gc({'nside': nside, 'zbin': 0,
                                 'data_catalogs': '/mnt/extraspace/damonge/S8z_data/DES_data/redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits',
        'file_mask': '/mnt/extraspace/damonge/S8z_data/DES_data/redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
        'file_nz': '/mnt/extraspace/damonge/S8z_data/DES_data/data_vector/2pt_NG_mcal_1110.fits'})

    eBOSS__0 = xcell.mappers.MappereBOSSQSO({'data_catalogs':
            ['/mnt/extraspace/gravityls_3/S8z/data/eBOSS_QSO/eBOSS_QSO_clustering_data-NGC-vDR16.fits',
             '/mnt/extraspace/gravityls_3/S8z/data/eBOSS_QSO/eBOSS_QSO_clustering_data-SGC-vDR16.fits'],
        'random_catalogs':
            ['/mnt/extraspace/gravityls_3/S8z/data/eBOSS_QSO/eBOSS_QSO_clustering_random-NGC-vDR16.fits',
             '/mnt/extraspace/gravityls_3/S8z/data/eBOSS_QSO/eBOSS_QSO_clustering_random-SGC-vDR16.fits'],
        'z_edges': [0, 1.5],
        'nside': nside})

    DECaLS = xcell.mappers.MapperDECaLS({
        'data_catalogs': [
            '/mnt/extraspace/damonge/S8z_data/DECALS/Legacy_Survey_BASS-MZLS_galaxies-selection.fits',
            '/mnt/extraspace/damonge/S8z_data/DECALS/Legacy_Survey_DECALS_galaxies-selection.fits'],
        'zbin': 0,
        'binary_mask': '/mnt/extraspace/damonge/S8z_data/DECALS/Legacy_footprint_final_mask_cut_decm36.fits',
        'nl_analytic': True,
        'completeness_map': '/mnt/extraspace/damonge/S8z_data/DECALS/Legacy_footprint_completeness_mask_128.fits',
        'star_map': '/mnt/extraspace/damonge/S8z_data/DECALS/allwise_total_rot_1024.fits',
        'nside': nside,
       }
    )

    K1000 = xcell.mappers.MapperKiDS1000({
        'data_catalog': '/mnt/extraspace/damonge/S8z_data/KiDS_1000/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits',
        'file_nz': '/mnt/extraspace/damonge/S8z_data/KiDS_1000/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO1_Nz.asc',
        'zbin': 0,
        'nside': nside,
        'path_lite': '/mnt/extraspace/gravityls_3/S8z/data/derived_products/K1000'
            })
    PLAcv = xcell.mappers.MapperP18CMBK({
      'file_klm': '/mnt/extraspace/damonge/S8z_data/Planck_data/COM_Lensing_4096_R3.00/MV/dat_klm.fits',
      'file_mask': '/mnt/extraspace/damonge/S8z_data/Planck_data/COM_Lensing_4096_R3.00/mask.fits.gz',
      'file_noise': '/mnt/extraspace/damonge/S8z_data/Planck_data/COM_Lensing_4096_R3.00/MV/nlkk.dat',
      'nside': nside,
      'path_lite': '/mnt/extraspace/damonge/S8z_data/Planck_data/COM_Lensing_4096_R3.00/MV/lite/'
    })

    ###### Option 1 ######
    eB0mask = scale_bin_map(eBOSS__0.get_mask(), -0.4)
    gc0mask = scale_bin_map(DESgc__0.get_mask(), -0.1)
    dec0mask = scale_bin_map(DECaLS.get_mask(), 0.1)
    k0mask = scale_bin_map(K1000.get_mask(), 0.4)

    mask = eB0mask + gc0mask + dec0mask + k0mask
    mask[mask == 0] = hp.UNSEEN

    hp.mollview(mask, cbar=False, title='', cmap='tab10', badcolor="lightgray")
    hp.graticule()

    plt.text(0., 0.50, 'eBOSS', fontsize=15, color='black', horizontalalignment='center')
    plt.text(0.80, 0.15, 'DELS', fontsize=15, color='black', horizontalalignment='left')
    plt.text(0.35, -0.47, 'KiDS', fontsize=15, color='black', horizontalalignment='left')
    plt.text(-0.1, -0.85, 'DES', fontsize=15, color='black', horizontalalignment='center')

    plt.savefig('footprint.pdf')
    plt.close()

    ###### Option 2 ######
    eB0mask = scale_bin_map(eBOSS__0.get_mask(), -0.2)
    gc0mask = scale_bin_map(DESgc__0.get_mask(), -0.1)
    dec0mask = scale_bin_map(DECaLS.get_mask(), 0.2)
    k0mask = scale_bin_map(K1000.get_mask(), 0.3)
    mask = dec0mask + gc0mask
    #
    eBpix = eB0mask != 0
    mask[eBpix] = eB0mask[eBpix]
    #
    kpix = k0mask != 0
    mask[kpix] = k0mask[kpix]
    #
    mask[mask == 0] = hp.UNSEEN

    hp.mollview(mask, cbar=False, title='', cmap='tab10', badcolor="lightgray")
    hp.graticule()

    plt.text(0., 0.50, 'eBOSS', fontsize=15, color='black', horizontalalignment='center')
    plt.text(0.80, 0.15, 'DELS', fontsize=15, color='black', horizontalalignment='left')
    plt.text(0.35, -0.47, 'KiDS', fontsize=15, color='black', horizontalalignment='left')
    plt.text(-0.1, -0.85, 'DES', fontsize=15, color='black', horizontalalignment='center')

    plt.savefig('footprint2.pdf')
    plt.close()

    ##### CMB ####
    cvmask = PLAcv.get_mask()
    mask = cvmask
    mask[mask == 0] = hp.UNSEEN
    hp.mollview(mask, cbar=False, title='', badcolor='lightgray')
    plt.savefig('footprint_cmbk.pdf')
    plt.close()
