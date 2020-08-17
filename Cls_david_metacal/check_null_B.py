import numpy as np
import scipy.stats as stats


def printflush(msg):
    print(msg)
    sys.stdout.flush()

nside = 4096
predir = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/'

pols = ['BB', 'EB', 'BE']
nmt_ids = [3, 1, 2]

for i1 in range(4):
    for i2 in range(i1, 4):
        fname_cls = predir + 'cls_metacal_cls_bins_%d%d_ns%d.npz' % (i1, i2, nside)
        fname_cov = predir + 'cls_metacal_covar_bins_'
        fname_cov += "new_nka_full_noise_"
        #fname_cov += "new_nka_"
        fname_cov += "%d%d_%d%d_ns%d.npz" % (i1, i2, i1, i2, nside)
        dcl = np.load(fname_cls)
        dcv = np.load(fname_cov)
        msk = dcl['ls'] < 2*nside

        if i1 == i2:
            ids = range(2)
        else:
            ids = range(3)

        print(i1, i2)
        for i in ids:
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
