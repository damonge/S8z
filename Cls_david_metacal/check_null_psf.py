import numpy as np
import scipy.stats as stats


def printflush(msg):
    print(msg)
    sys.stdout.flush()

nside = 4096
predir = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/'

pols = ['EE', 'EB', 'BE', 'BB']
nmt_ids = range(4)

for i1 in range(4):
    fname_cls = predir + 'cls_metacal_cls_bins_%d%d_ns%d_xpsf.npz' % (i1, i1, nside)
    fname_cov = predir + 'cls_metacal_covar_xpsf_bin%d_ns%d.npz' % (i1, nside)
    dcl = np.load(fname_cls)
    dcv = np.load(fname_cov)
    msk = dcl['ls'] < 2*nside

    print(i1)
    for i in nmt_ids:
        j = nmt_ids[i]
        d = (dcl['cls']-dcl['nls'])[j][msk]
        cv = dcv['cov'][:, j, :, j][msk][:, msk]
        chi2 = np.dot(d, np.linalg.solve(cv, d))
        ndof = len(d)
        print(pols[i] + ':')
        print("  chi2 = %.3lf, " % chi2 +
              "ndof = %d, " % ndof +
              "p=%.1lE" % (1-stats.chi2.cdf(chi2, ndof)))
        print(" ")
