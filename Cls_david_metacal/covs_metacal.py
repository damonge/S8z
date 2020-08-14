import numpy as np
import healpy as hp
from argparse import ArgumentParser
import pymaster as nmt
import os
import sys
import pyccl as ccl


def printflush(msg):
    print(msg)
    sys.stdout.flush()


parser = ArgumentParser()
parser.add_argument("--bin-a1", default=0, type=int, help="Bin number")
parser.add_argument("--bin-a2", default=0, type=int, help="Bin number")
parser.add_argument("--bin-b1", default=0, type=int, help="Bin number")
parser.add_argument("--bin-b2", default=0, type=int, help="Bin number")
parser.add_argument("--nside", default=4096, type=int, help="Nside")
parser.add_argument("--n-iter", default=0, type=int, help="n_iter")
parser.add_argument("--recompute-mcm", default=False, action='store_true',
                    help="Recompute MCM even if it exists?")
parser.add_argument("--old-nka", default=False, action='store_true',
                    help="Use old NKA")
o = parser.parse_args()


predir = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/'

npix = hp.nside2npix(o.nside)


printflush("Theory spectra")
ls = np.arange(3*o.nside)
cl0 = np.zeros(3*o.nside)

cosmo = ccl.Cosmology(Omega_c=0.260-0.0479,
                      Omega_b=0.0479,
                      h=0.685,
                      n_s=0.973,
                      sigma8=0.821)


def get_field(bin_no, return_mask=False):
    prefix = predir + "maps_metacal_bin%d_ns%d" % (bin_no, o.nside)
    pix = np.load(prefix + '_goodpix.npz')['pix']
    wmap = np.zeros(npix)
    wmap[pix] = np.load(prefix + '_w.npz')['w']
    if return_mask:
        return wmap
    return nmt.NmtField(wmap, [wmap, wmap], n_iter=o.n_iter)


def get_tracer(bin_no):
    fname_nz = predir + "dndz_metacal_bin%d.txt" % bin_no
    zi, zm, zf, dndz = np.loadtxt(fname_nz, unpack=True)
    tr = ccl.WeakLensingTracer(cosmo, (zi, dndz))
    return tr

tracers = {}
if o.bin_a1 not in tracers:
    tracers[o.bin_a1] = get_tracer(o.bin_a1)
if o.bin_a2 not in tracers:
    tracers[o.bin_a2] = get_tracer(o.bin_a2)
if o.bin_b1 not in tracers:
    tracers[o.bin_b1] = get_tracer(o.bin_b1)
if o.bin_b2 not in tracers:
    tracers[o.bin_b2] = get_tracer(o.bin_b2)


def get_cl(trs, b1, b2):
    nl = np.zeros(3*o.nside)
    if b1 == b2:
        fname_nl = predir + "maps_metacal_bin%d_ns%d_nells.npz" % (b1, o.nside)
        d = np.load(fname_nl)
        nl[:2] = d['nl_cov']
    sl = ccl.angular_cl(cosmo, trs[b1], trs[b2], ls)
    if o.old_nka:
        return np.array([sl + nl, cl0, cl0, nl])
    else:
        w = nmt.NmtWorkspace()
        predir_mcm = predir + 'cls_metacal_mcm_bins_'
        fname_mcm = predir_mcm + '%d%d_ns%d.fits' % (b1, b2, o.nside)
        if os.path.isfile(fname_mcm):
            w.read_from(fname_mcm)
        else:
            fname_mcm = predir_mcm + '%d%d_ns%d.fits' % (b2, b1, o.nside)
            if os.path.isfile(fname_mcm):
                w.read_from(fname_mcm)
            else:
                raise ValueError("Can't find MCM " + fname_mcm)
        mskprod = get_field(b1, return_mask=True)
        if b1 == b2:
            mskprod *= mskprod
        else:
            mskprod *= get_field(b2, return_mask=True)
        fsky = np.mean(mskprod)
        return w.couple_cell([sl, cl0, cl0, cl0])/fsky + np.array([nl, cl0, cl0, nl])

clt = {}
k = '%d%d' % (o.bin_a1, o.bin_b1)
if k not in clt:
    clt[k] = get_cl(tracers, o.bin_a1, o.bin_b1)
k = '%d%d' % (o.bin_a1, o.bin_b2)
if k not in clt:
    clt[k] = get_cl(tracers, o.bin_a1, o.bin_b2)
k = '%d%d' % (o.bin_a2, o.bin_b1)
if k not in clt:
    clt[k] = get_cl(tracers, o.bin_a2, o.bin_b1)
k = '%d%d' % (o.bin_a2, o.bin_b2)
if k not in clt:
    clt[k] = get_cl(tracers, o.bin_a2, o.bin_b2)


printflush("CMCM")
fname_cmcm = predir + 'cls_metacal_cmcm_bins_'
fname_cmcm += '%d%d_%d%d_ns%d.fits' % (o.bin_a1, o.bin_a2, o.bin_b1, o.bin_b2, o.nside)

cw = nmt.NmtCovarianceWorkspace()
if os.path.isfile(fname_cmcm) and not o.recompute_mcm:
    printflush(" - Reading")
    cw.read_from(fname_cmcm)
else:
    printflush(" - Fields")
    fields = {}
    if o.bin_a1 not in fields:
        fields[o.bin_a1] = get_field(o.bin_a1)
    if o.bin_a2 not in fields:
        fields[o.bin_a2] = get_field(o.bin_a2)
    if o.bin_b1 not in fields:
        fields[o.bin_b1] = get_field(o.bin_b1)
    if o.bin_b2 not in fields:
        fields[o.bin_b2] = get_field(o.bin_b2)
    printflush(" - Computing")
    cw.compute_coupling_coefficients(fields[o.bin_a1], fields[o.bin_a2],
                                     fields[o.bin_b1], fields[o.bin_b2])
    cw.write_to(fname_cmcm)


printflush("MCMs")
fname_mcm_a = predir + 'cls_metacal_mcm_bins_'
fname_mcm_a += '%d%d_ns%d.fits' % (o.bin_a1, o.bin_a2, o.nside)
wa = nmt.NmtWorkspace()
wa.read_from(fname_mcm_a)
if (o.bin_a1 == o.bin_b1) and (o.bin_a2 == o.bin_b2):
    wb = wa
else:
    fname_mcm_b = predir + 'cls_metacal_mcm_bins_'
    fname_mcm_b += '%d%d_ns%d.fits' % (o.bin_b1, o.bin_b2, o.nside)
    wb = nmt.NmtWorkspace()
    wb.read_from(fname_mcm_b)
nbpw = wa.wsp.bin.n_bands


printflush("Covariance")
cov = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                              clt['%d%d' % (o.bin_a1, o.bin_b1)],
                              clt['%d%d' % (o.bin_a1, o.bin_b2)],
                              clt['%d%d' % (o.bin_a2, o.bin_b1)],
                              clt['%d%d' % (o.bin_a2, o.bin_b2)],
                              wa, wb).reshape([nbpw, 4, nbpw, 4])

printflush("Writing")
fname_cov = predir + 'cls_metacal_covar_bins_'
if not o.old_nka:
    fname_cov += "new_nka_"
fname_cov += '%d%d_%d%d_ns%d.npz' % (o.bin_a1, o.bin_a2, o.bin_b1, o.bin_b2, o.nside)
np.savez(fname_cov, cov=cov)
