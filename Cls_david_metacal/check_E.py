import numpy as np
import scipy.stats as stats
import pyccl as ccl
import pymaster as nmt
import os


def printflush(msg):
    print(msg)
    sys.stdout.flush()

nside = 4096
predir = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/'


cosmo = ccl.Cosmology(Omega_c=0.260-0.0479,
                      Omega_b=0.0479,
                      h=0.685,
                      n_s=0.973,
                      sigma8=0.821)

def get_tracer(bin_no):
    fname_nz = predir + "dndz_metacal_bin%d.txt" % bin_no
    zi, zm, zf, dndz = np.loadtxt(fname_nz, unpack=True)
    tr = ccl.WeakLensingTracer(cosmo, (zi, dndz))
    return tr

trs = [get_tracer(i) for i in range(4)]
ls = np.arange(3*nside)
cl0 = np.zeros(3*nside)

fname_cls0 = predir + 'cls_metacal_cls_bins_00_ns%d.npz' % nside
leff = np.load(fname_cls0)['ls']
msk = leff < 2*nside
nbpw = np.sum(msk)

dls = np.zeros([4, 4, nbpw])
tls = np.zeros([4, 4, nbpw])
for i1 in range(4):
    for i2 in range(i1, 4):
        print(i1, i2)
        fname_tls = predir + "cls_metacal_tls_%d%d_ns%d.npz" % (i1, i2, nside)
        tl = ccl.angular_cl(cosmo, trs[i1], trs[i2], ls)
        fname_win = predir + "cls_metacal_win_bins_%d%d_ns%d.npz" % (i1, i2, nside)
        win = np.load(fname_win)['win'][0, :, 0, :]
        tl = np.dot(win, tl)

        fname_cls = predir + 'cls_metacal_cls_bins_%d%d_ns%d.npz' % (i1, i2, nside)
        fname_cov = predir + 'cls_metacal_covar_bins_'
        fname_cov += "new_nka_full_noise_"
        fname_cov += "%d%d_%d%d_ns%d.npz" % (i1, i2, i1, i2, nside)
        dcl = np.load(fname_cls)
        dcv = np.load(fname_cov)

        tl = tl[msk]
        dl = (dcl['cls']-dcl['nls'])[0][msk]
        rl = dl-tl
        cv = dcv['cov'][:, 0, :, 0][msk][:, msk]
        chi2 = np.dot(rl, np.linalg.solve(cv, rl))
        ndof = len(rl)
        print("  chi2 = %.3lf, " % chi2 +
              "ndof = %d, " % ndof +
              "p=%.1lE" % (1-stats.chi2.cdf(chi2, ndof)))
        print(" ")

        tls[i1, i2, :] = tl
        dls[i1, i2, :] = dl
        if i1 != i2:
            tls[i2, i1, :] = tl
            dls[i2, i1, :] = dl

print("Combined:")
ncls = (4 * (4+1)) // 2
ndata = ncls * nbpw
i = 0
t = np.zeros([ncls, nbpw])
d = np.zeros([ncls, nbpw])
c = np.zeros([ncls, nbpw, ncls, nbpw])
for i1 in range(4):
    for i2 in range(i1, 4):
        j = 0
        t[i, :] = tls[i1, i2]
        d[i, :] = dls[i1, i2]
        for j1 in range(4):
            for j2 in range(j1, 4):
                if j < i:
                    j += 1
                    continue
                fname_cov = predir + 'cls_metacal_covar_bins_'
                fname_cov += "new_nka_full_noise_"
                fname_cov += "%d%d_%d%d_ns%d.npz" % (i1, i2, j1, j2, nside)
                cov = np.load(fname_cov)['cov']
                cov = cov[:, 0, :, 0]
                cov = cov[msk][:, msk]
                c[i, :, j, :] = cov
                if i != j:
                    c[j, :, i, :] = cov.T
                j += 1
        i += 1
t = t.flatten()
d = d.flatten()
c = c.reshape([ncls * nbpw, ncls * nbpw])
r = d-t
ndof = len(r)
chi2 = np.dot(r, np.linalg.solve(c, r))
print("  chi2 = %.3lf, " % chi2 +
      "ndof = %d, " % ndof +
      "p=%.1lE" % (1-stats.chi2.cdf(chi2, ndof)))
print(" ")
