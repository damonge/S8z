#!/usr/bin/python
import getdist
import numpy as np
import os
import pyccl as ccl

MP_root = '/mnt/zfsusers/gravityls_3/codes/montepython_emilio'


des_root = os.path.join(MP_root, 'chains/cl_cross_corr_des_full_kappa_l8/')
des = getdist.loadMCSamples(os.path.join(des_root, '2020-03-09_1000000_'),
                           settings={'ignore_rows':0.1})

growth_root = os.path.join(MP_root, 'chains/cl_cross_corr_des_growth_dpk0_fixed_kappa_l8/')
growth = getdist.loadMCSamples(os.path.join(growth_root, '2020-03-09_1000000_'),
                           settings={'ignore_rows':0.1})


def S8z(mcmc, a, MG=False, size=None):
    p = mcmc.getParams()
    if size is None:
        size = p.A_s.size
    S8_ar = np.zeros((size, a.size))

    for i in range(size):
        cosmo = ccl.Cosmology(h=p.h[i], Omega_c=p.Omega_c[i], Omega_b=p.Omega_b[i],
                                A_s=1e-9*p.A_s[i], n_s=p.n_s[i],
                                w0=-1, wa=0,
                                transfer_function='boltzmann_class')
        # Compute everything
        sigma8 = ccl.sigma8(cosmo)
        Dz = ccl.background.growth_factor(cosmo, a)
        Om = ccl.background.omega_x(cosmo, 1, 'matter')

        if MG:
            d1 = p.dpk1[i]
        else:
            d1 = 0
            
        S8_ar[i] = (1 + d1 * (1 - a)) * Dz * sigma8 * (Om / 0.3) ** 0.5
        
    return S8_ar


a = np.logspace(0, -1, 100)

np.savez_compressed(os.path.join(des_root, 'S8z.npz'), S8z=S8z(des, a), a=a)
np.savez_compressed(os.path.join(growth_root, 'S8z.npz'), S8z=S8z(growth, a), a=a)
