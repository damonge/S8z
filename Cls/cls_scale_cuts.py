#!/usr/bin/python

import numpy as np
import os
import pyccl as ccl
import yaml

def read_yml(file):
    with open(file, 'r') as stream:
        return yaml.safe_load(stream)

def write_yml(yml, file, header=''):
    with open(file, 'w') as stream:
        stream.write(header)
        stream.write('\n')
        yaml.dump(yml, stream)

def get_cosmo_ccl(cosmo_params, baryons=False):
    if baryons:
        baryons_power_spectrum = "bcm"
    else:
        baryons_power_spectrum = "nobaryons"

    cosmo = ccl.Cosmology(
        h        = cosmo_params['h'],
        Omega_c  = cosmo_params['Omega_c'],
        Omega_b  = cosmo_params['Omega_b'],
        sigma8   = cosmo_params['sigma8'],
        # A_s      = cosmo_params['A_s'],
        n_s      = cosmo_params['n_s'],
        w0       = cosmo_params['w0'],
        wa       = cosmo_params['wa'],
        transfer_function = 'boltzmann_class',
        baryons_power_spectrum = baryons_power_spectrum
    )

    return cosmo

def get_tracers_ccl(cosmo, tracers_info):
    # Get Tracers
    for tr in tracers_info['maps']:
        if tr['type'] == 'gc':
            # Import z, pz
            fname = os.path.join(files_root, tr['dndz_file'])
            z, pz = np.loadtxt(fname, usecols=(1,3), unpack=True)
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
            # Calculate bias
            bias = tr['bias']
            bz = bias*np.ones(z.shape)
            # Get tracer
            tr['tracer'] = ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z_dz,pz),bias=(z,bz))
        elif tr['type'] == 'wl':
            # Import z, pz
            fname = os.path.join(files_root, tr['dndz_file'])
            z, pz = np.loadtxt(fname, usecols=(1,3), unpack=True)
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
            # # Calculate bias IA
            # A =
            # eta =
            # z0 =
            # bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474  # pyccl2 -> has already the factor inside. Only needed bz
            # Get tracer
            tr['tracer'] = ccl.WeakLensingTracer(cosmo,dndz=(z_dz,pz)) # ,ia_bias=(z,bz))
        elif tr['type'] == 'cv':
            tr['tracer'] = ccl.CMBLensingTracer(cosmo, z_source=1100)#TODO: correct z_source
        else:
            raise ValueError('Type of tracer not recognized. It can be gc, wl or cv!')

def compute_cls(cosmo, tracers_info, ells, outdir):
    # Get the Tracer instaces
    print('Get tracer instances')
    get_tracers_ccl(cosmo, tracers_info)

    cls_ar = np.zeros((len(tracers_info['data_vectors']), ells.size))
    # Get theory Cls
    print('Computing theory Cls')
    for i, dv in enumerate(tracers_info['data_vectors']):
        tracer1 = next(x['tracer'] for x in tracers_info['maps'] if x['name']==dv['tracers'][0])
        tracer2 = next(x['tracer'] for x in tracers_info['maps'] if x['name']==dv['tracers'][1])
        cls = ccl.angular_cl(cosmo, tracer1, tracer2, ells)
        # # Add multiplicative bias to WL
        # type1 = next(x['type'] for x in tracers_info['maps'] if x['name']==dv['tracers'][0])
        # type2 = next(x['type'] for x in tracers_info['maps'] if x['name']==dv['tracers'][1])
        # if type1 == 'wl':
        #     bin = next(x['bin'] for x in tracers_info['maps'] if x['name']==dv['tracers'][0])
        #     m =
        #     cls = (1.+m)*cls
        # if type2 == 'wl':
        #     bin = next(x['bin'] for x in tracers_info['maps'] if x['name']==dv['tracers'][1])
        #     m =
        #     cls = (1.+m)*cls

        fname = os.path.join(outdir, 'cls_{}_{}.npz'.format(*dv['tracers']))
        print ('Saving {}'.format(fname))
        np.savez_compressed(fname, cls=cls, ells=ells)
        cls_ar[i] = cls
    return cls_ar

def get_ell_from_k(cosmo, k, a):
    return k * ccl.comoving_radial_distance(cosmo, a)

def estimate_ell_cuts_shear(ell, cls_bcm, cls_nob, maxreldev):
    print('Finding shear ell_cut with rdev = baryons / nobaryons - 1 = {}'.format(maxreldev))
    rdev = np.abs(cls_bcm / cls_nob - 1)
    ell_cuts_index = np.nanargmax(rdev > maxreldev, axis=-1)

    return ell[ell_cuts_index]

def estimate_ell_cuts_clustering(cosmo, kmax, z):
    # z can be a float or an array
    print('Finding clustering ell_cut with kmax = {} at z = {}'.format(kmax, z))
    return get_ell_from_k(cosmo, kmax, 1/(1+z))

def estimate_ell_cuts(ells, cls_bcm, cls_nob, cosmo, tracers_info,
                      files_root, kmax=0.1, maxreldev=0.02):
    h = cosmo.cosmo.params.h  # k in units of Mpc^-1 / h

    zbin = []
    for tr in tracers_info['maps']:
        if tr['type'] != 'gc':
            continue
        fname = os.path.join(files_root, tr['dndz_file'])
        z, pz = np.loadtxt(fname, usecols=(1,3), unpack=True)
        zbin.append(np.sum(z * pz) / np.sum(pz))

    ell_cuts = np.ones(len(tracers_info['data_vectors'])) * ells[-1]
    for i, dv in enumerate(tracers_info['data_vectors']):
        print dv
        for tr in dv['tracers']:
            if 'gc' in tr:
                ell_tmp = estimate_ell_cuts_clustering(cosmo, kmax/h, zbin[int(tr[-1])])
            elif 'wl' in tr:
                ell_tmp = estimate_ell_cuts_shear(ells, cls_bcm[i], cls_nob[i],
                                                  maxreldev)
            elif 'cv' in tr:
                ell_tmp = 2000 # ells[-1]

            print ell_tmp

            ell_cuts[i] = np.min([ell_cuts[i], ell_tmp])

    return ell_cuts



#############################################################################

def main(files_root, outdir, baryons):
    cosmo_file = os.path.join(files_root, "fiducial_cosmology.yml")
    tracers_file = os.path.join(files_root, "tracers_info.yml")
    bpw_file = os.path.join(files_root, 'bandpower_intervals.txt')

    ##### Load fiducial, tracers and ell files
    print('Loading tracer_info')
    tracers_info = read_yml(tracers_file)
    print('Loading cosmo_params')
    cosmo_params = read_yml(cosmo_file)['cosmo_params']
    cosmo_params['Omega_c'] = cosmo_params['Omega_m'] - cosmo_params['Omega_b']
    cosmo_params['w0'] = -1
    cosmo_params['wa'] = 0
    print('Loading ells')
    ells = np.arange(np.loadtxt(bpw_file)[-1])
    ######

    if baryons:
        outdir = os.path.join(outdir, 'bcm')
    else:
        outdir = os.path.join(outdir, 'nobaryons')

    # Save a copy of the yml files
    fname = os.path.join(outdir, 'tracers_info.yml')
    print('Writing tracer file in {}'.format(fname))
    write_yml(tracers_info, fname)
    fname = os.path.join(outdir, 'fiducial_cosmology.yml')
    print('Writing fiducial cosmo file in {}'.format(fname))
    write_yml({'cosmo_params': cosmo_params}, fname)
    #

    print('Generating the ccl cosmo instance')
    cosmo = get_cosmo_ccl(cosmo_params, baryons)
    print('Computing cls')
    return ells, compute_cls(cosmo, tracers_info, ells, outdir), cosmo
    ######

#############################################################################
#############################################################################

# files_root = "/mnt/zfsusers/gravityls_3/codes/S8z/NGC/"
outdir = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial-5pc/'
files_root = outdir
maxreldev = 0.05
kmax = 0.1
header='# ell_cuts calculated as the min ell for which baryons/nobaryons -1 > {} or kmax = {}\n'.format(maxreldev, kmax)

#######

print('BCM')
ells, cls_bcm, _ = main(files_root, outdir, baryons=True)
print('No Baryons')
ells, cls_nob, cosmo_nob = main(files_root, outdir, baryons=False)

#######
print('Loading tracers_info to add ell_cuts')
fname = os.path.join(outdir, 'nobaryons',  'tracers_info.yml')
tracers_info = read_yml(fname)
##
ell_cuts = estimate_ell_cuts(ells, cls_bcm, cls_nob, cosmo_nob, tracers_info, files_root,
                             kmax=kmax, maxreldev=maxreldev)
for i, tr in enumerate(tracers_info['data_vectors']):
    lmin = 0
    if 'PLAcv' in tr['tracers']:
        lmin = 8

    tr['ell_cuts'] = [lmin,  int(ell_cuts[i])]  # For yml compatibility

##
fname = os.path.join(outdir, 'tracers_info_with_ell_cuts.yml')
print('Saving tracers_info with add ell_cuts in {}'.format(fname))
write_yml(tracers_info, fname, header)
print('Finished')
