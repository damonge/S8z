import numpy as np
from argparse import ArgumentParser
import sacc


parser = ArgumentParser()
parser.add_argument("--nside", default=4096, type=int, help="Nside")
o = parser.parse_args()


predir = '/mnt/extraspace/damonge/S8z_data/'
predir += 'derived_products/des_shear/'

ls_large = np.arange(3*o.nside)
nbins = 4
nmaps = nbins * 2
ncross = (nmaps * (nmaps + 1)) // 2
nbpw = len(np.load(predir + f'cls_metacal_cls_bins_00_ns{o.nside}.npz')['ls'])
ndata = ncross * nbpw
pname = ['e', 'b']


def sacc_and_tracers():
    # Initializes a sacc file and fills it up with the shear tracers
    s = sacc.Sacc()
    for b in range(nbins):
        z0, zm, zf, nz = np.loadtxt(predir + f'dndz_metacal_bin{b}.txt',
                                    unpack=True)
        s.add_tracer('NZ', f'bin{b}',
                     quantity='galaxy_shear',
                     spin=2,
                     z=zm,
                     nz=nz,
                     extra_columns={'z_i': z0,
                                    'z_f': zf})
    return s


def pol_pair_iterator(b1, b2):
    # Iterates over E/B pair combinations
    for p1 in range(2):
        if b1 == b2:
            p2_range = range(p1, 2)
        else:
            p2_range = range(2)
        for p2 in p2_range:
            ipx = p2 + 2 * p1
            yield p1, p2, ipx


def bin_pair_iterator():
    # Iterates over bin pair combinations
    ibx = 0
    for b1 in range(nbins):
        for b2 in range(b1, nbins):
            yield b1, b2, ibx
            ibx += 1


# SACC with signal + Gaussian covariance
# Initialize and fill tracers
s1 = sacc_and_tracers()
# Add power spectra
ix = 0
for b1, b2, ibx in bin_pair_iterator():
    fname_cl = predir + f'cls_metacal_cls_bins_{b1}{b2}_ns{o.nside}.npz'
    fname_win = predir + f'cls_metacal_win_bins_{b1}{b2}_ns{o.nside}.npz'
    dc = np.load(fname_cl)
    ls = dc['ls']
    cl = dc['cls'] - dc['nls']
    w = np.load(fname_win)['win']
    n1 = f'bin{b1}'
    n2 = f'bin{b2}'
    for p1, p2, ipx in pol_pair_iterator(b1, b2):
        cl_h = cl[ipx, :]
        w_h = sacc.BandpowerWindow(ls_large, w[ipx, :, ipx, :].T)
        s1.add_ell_cl('cl_' + pname[p1] + pname[p2],
                      n1, n2, ls, cl_h,
                      window=w_h)
        ix += 1
# Add covariance matrix
prefix_cov = predir + 'cls_metacal_covar_bins_new_nka_full_noise'
cov = np.zeros([ncross, nbpw, ncross, nbpw])
ix_a = 0
for b1_a, b2_a, ibx_a in bin_pair_iterator():
    covs = {}
    # Read all possible covariance elements
    for b1_b, b2_b, ibx_b in bin_pair_iterator():
        if ibx_b >= ibx_a:
            fname_cov = prefix_cov + f'_{b1_a}{b2_a}'
            fname_cov += f'_{b1_b}{b2_b}_ns{o.nside}.npz'
            covs[f'{b1_b}{b2_b}'] = np.load(fname_cov)['cov']
    # Loop over polarizations
    for p1_a, p2_a, ipx_a in pol_pair_iterator(b1_a, b2_a):
        ix_b = 0
        # Loop over possible cross-correlations
        for b1_b, b2_b, ibx_b in bin_pair_iterator():  
            for p1_b, p2_b, ipx_b in pol_pair_iterator(b1_b, b2_b):
                if ix_b >= ix_a:  # Skip if below the diagonal
                    cv = covs[f'{b1_b}{b2_b}'][:, ipx_a, :, ipx_b]
                    cov[ix_a, :, ix_b, :] = cv
                    if ix_a != ix_b: 
                        cov[ix_b, :, ix_a, :] = cv.T
                ix_b += 1
        ix_a += 1
cov = cov.reshape([ncross * nbpw, ncross * nbpw])
s1.add_covariance(cov)
s1.save_fits(predir + f'cls_signal_covG_ns{o.nside}.fits',
             overwrite=True)


# SACC with noise + non-Gaussian covariance
# Initialize and fill tracers
s2 = sacc_and_tracers()
# Add power spectra
for b1, b2, ibx in bin_pair_iterator():
    fname_cl = predir + f'cls_metacal_cls_bins_{b1}{b2}_ns{o.nside}.npz'
    dc = np.load(fname_cl)
    ls = dc['ls']
    nl = dc['nls']
    n1 = f'bin{b1}'
    n2 = f'bin{b2}'
    for p1, p2, ipx in pol_pair_iterator(b1, b2):
        nl_h = nl[ipx, :]
        s2.add_ell_cl('cl_' + pname[p1] + pname[p2],
                      n1, n2, ls, nl_h)
# Read full non-Gaussian covariance
cov_ng = np.load('/mnt/extraspace/damonge/S8z_data/derived_products/'
                 'covmat_zeroed_g_gc_cmbk_s8z_NG+SSC.npy').reshape([44, nbpw, 44, nbpw])
cov_ng = cov_ng[:10, :, :, :][:, :, :10, :]
cov = np.zeros([ncross, nbpw, ncross, nbpw])
cv0 = np.zeros([nbpw, nbpw])
ix_a = 0
for b1_a, b2_a, ibx_a in bin_pair_iterator():
    for p1_a, p2_a, ipx_a in pol_pair_iterator(b1_a, b2_a):
        ix_b = 0
        for b1_b, b2_b, ibx_b in bin_pair_iterator():  
            for p1_b, p2_b, ipx_b in pol_pair_iterator(b1_b, b2_b):
                if ix_b >= ix_a:  # Skip if below the diagonal
                    if p1_b == p2_b == p1_a == p2_a == 0:
                        cv = cov_ng[ibx_a, :, ibx_b, :]
                    else:
                        cv = cv0
                    cov[ix_a, :, ix_b, :] = cv
                    if ix_a != ix_b: 
                        cov[ix_b, :, ix_a, :] = cv.T
                ix_b += 1
        ix_a += 1
cov = cov.reshape([ncross * nbpw, ncross * nbpw])
s2.add_covariance(cov)
s2.save_fits(predir + f'cls_noise_covNG_ns{o.nside}.fits',
             overwrite=True)


# SACC with PSF power spectra
# Initialize and fill tracers
s3 = sacc_and_tracers()
# Add PSF tracers
for b in range(nbins):
    s3.add_tracer('Misc', f'psf{b}')
# Add spectra
for b in range(nbins):
    fname_cl = predir + f'cls_metacal_cls_bins_00_ns4096_xpsf.npz'
    dc = np.load(fname_cl)
    ls = dc['ls']
    cl = dc['cls']
    for p1, p2, ipx in pol_pair_iterator(0, 1):
        cl_h = cl[ipx, :]
        s3.add_ell_cl('cl_' + pname[p1] + pname[p2],
                      f'bin{b}', f'psf{b}', ls, cl_h)
# Add covar
prefix_cov = predir + 'cls_metacal_covar_xpsf_'
cov = np.zeros([nbins * 4, nbpw, nbins * 4, nbpw])
ix_a = 0
for b_a in range(nbins):
    for p1_a, p2_a, ipx_a in pol_pair_iterator(0, 1):
        ix_b = 0
        for b_b in range(nbins):
            if b_b == b_a:
                fname_cov = prefix_cov + f'bin{b_a}_ns{o.nside}.npz'
                cv = np.load(fname_cov)['cov']
            else:
                cv = None
            for p1_b, p2_b, ipx_b in pol_pair_iterator(0, 1):
                if b_b == b_a:
                    cov[ix_a, :, ix_b, :] = cv[:, ipx_a, :, ipx_b]
                ix_b += 1
        ix_a += 1
cov = cov.reshape([nbins * 4 * nbpw, nbins * 4 * nbpw])
s3.add_covariance(cov)
s3.save_fits(predir + f'cls_xpsf_ns{o.nside}.fits',
             overwrite=True)
