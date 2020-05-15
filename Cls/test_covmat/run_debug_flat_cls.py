from __future__ import print_function
from optparse import OptionParser
from scipy.interpolate import interp1d
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--outdir',dest='outdir',default='./sims_debug_flat_cls',type=str,
                  help='Output directory')
parser.add_option('--nside', dest='nside', default=4096, type=int,
                  help='HEALPix nside param')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')

(o, args) = parser.parse_args()

##############################################################################
# Create outdir
os.makedirs(o.outdir, exist_ok=True)
# Set files prefix
prefix_out = os.path.join(o.outdir, 'run_debug_flat_cls_')
# Set nside
valid_nside = [2048, 4096]
if not o.nside in valid_nside:
    raise ValueError('nside must be one of {}'.format(valid_nside))
nside = o.nside  # 2048
# Set nsims
nsims=o.isim_end-o.isim_ini+1
# Set root path of observations
obs_path = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together'
if nside != 4096:
    obs_path += '_{}'.format(nside)
##############################################################################

##############################################################################
#Read input power spectra
# gc3 - wl1
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc3_DESwl1.npz'
fid_data = np.load(fname)
l, clte = fid_data['ells'][:3*nside], fid_data['cls'][:3*nside]
# gc3 - gc3
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc3_DESgc3.npz'
cltt = np.load(fname)['cls'][:3*nside]
# wl1 - wl1
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESwl1_DESwl1.npz'
clee = np.load(fname)['cls'][:3*nside]
#
cltb = clbb = clbe = cleb = 0 * cltt

# Read noise
nltt = 0*cltt
nlte = 0*cltt
nlee = 0*cltt
nlbb = 0*cltt

# Make them flat
cltt = np.ones(l.size)
clee = np.ones(l.size)
clbb = np.ones(l.size)

#Read mask
fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
mask_gc = hp.read_map(fname, verbose=False)

fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/map_metacal_bin1_counts_w_ns{}.fits'.format(nside)
mask_wl = hp.read_map(fname, verbose=False)

#Read bpw
fname = os.path.join(obs_path, 'l_bpw.txt')
lbpw = np.loadtxt(fname)
##############################################################################


#Generate an initial simulation
def get_fields() :
    """
    Generate a simulated field.
    It returns two NmtField objects for a spin-0 and a spin-2 field.

    :param fsk: a fm.FlatMapInfo object.
    :param mask: a sky mask.
    :param w_cont: deproject any contaminants? (not implemented yet)
    """
    st,sq,su=hp.synfast([cltt+nltt,clee+nlee,clbb+nlbb,clte+nlte],o.nside,new=True,verbose=False,pol=True,n_iter=0)
    ff0=nmt.NmtField(mask_gc,[st])
    ff2=nmt.NmtField(mask_wl,[sq, su])
    return ff0, ff2

np.random.seed(1000)
f0, f2 = get_fields()

##############################################################################
#Compute mode-coupling matrix
##############################################################################
w00 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w00_00.dat") : #spin0-spin2
    print("Computing w00")
    w00.compute_coupling_matrix(f0,f0,b, n_iter=0)
    w00.write_to(prefix_out+"_w00_00.dat");
else:
    w00.read_from(prefix_out+"_w00_00.dat")

w02 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w02_02.dat") : #spin0-spin2
    print("Computing w02")
    w02.compute_coupling_matrix(f0,f2,b, n_iter=0)
    w02.write_to(prefix_out+"_w02_02.dat");
else:
    w02.read_from(prefix_out+"_w02_02.dat")

w22 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w22_22.dat") : #spin0-spin2
    print("Computing w22")
    w22.compute_coupling_matrix(f2,f2,b, n_iter=0)
    w22.write_to(prefix_out+"_w22_22.dat");
else:
    w22.read_from(prefix_out+"_w22_22.dat")

##############################################################################
# Compute covariance (only depends on the masks)
##############################################################################
if not os.path.isfile(prefix_out + '_covTh.npz'):
    cw00_00 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0000.dat") : # mask0-mask0-mask0-mask0
        print("Computing cw0000")
        cw00_00.compute_coupling_coefficients(f0, f0, f0, f0, n_iter=0)
        cw00_00.write_to(prefix_out + "_cw0000.dat")
    else:
        cw00_00.read_from(prefix_out + '_cw0000.dat')

    cw00_02 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0002.dat") : # mask0-mask0-mask0-mask2
        print("Computing cw0002")
        cw00_02.compute_coupling_coefficients(f0, f0, f0, f2, n_iter=0)
        cw00_02.write_to(prefix_out + "_cw0002.dat")
    else:
        cw00_02.read_from(prefix_out + '_cw0002.dat')

    cw00_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0022.dat") : # mask0-mask0-mask2-mask2
        print("Computing cw0022")
        cw00_22.compute_coupling_coefficients(f0, f0, f2, f2, n_iter=0)
        cw00_22.write_to(prefix_out + "_cw0022.dat")
    else:
        cw00_22.read_from(prefix_out + '_cw0022.dat')

    cw02_02 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0202.dat") : # mask0-mask2-mask0-mask2
        print("Computing cw0202")
        cw02_02.compute_coupling_coefficients(f0, f2, f0, f2, n_iter=0)
        cw02_02.write_to(prefix_out + "_cw0202.dat")
    else:
        cw02_02.read_from(prefix_out + '_cw0202.dat')

    cw02_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0222.dat") : # mask0-mask2-mask2-mask2
        print("Computing cw0222")
        cw02_22.compute_coupling_coefficients(f0, f2, f2, f2, n_iter=0)
        cw02_22.write_to(prefix_out + "_cw0222.dat")
    else:
        cw02_22.read_from(prefix_out + '_cw0222.dat')

    cw22_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw2222.dat") : # mask2-mask2-mask2-mask2
        print("Computing cw2222")
        cw22_22.compute_coupling_coefficients(f2, f2, f2, f2, n_iter=0)
        cw22_22.write_to(prefix_out + "_cw2222.dat")
    else:
        cw22_22.read_from(prefix_out + '_cw2222.dat')

    ##
    # Compute matrices
    ##
    covar_00_00 = nmt.gaussian_covariance(cw00_00,
                                          0, 0, 0, 0,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          w00, wb=w00).reshape([n_ell, 1,
                                                                n_ell, 1])

    covar_00_02 = nmt.gaussian_covariance(cw00_02, 0, 0, 0, 2,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb]  # TE, TB
                                          w00, wb=w02).reshape([n_ell, 1,
                                                                n_ell, 2])

    covar_00_22 = nmt.gaussian_covariance(cw00_22, 0, 0, 2, 2,  # Spins of the 4 fields
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          w00, wb=w22).reshape([n_ell, 1,
                                                                n_ell, 4])

    covar_02_02 = nmt.gaussian_covariance(cw02_02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # ET, BT
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          w02, wb=w02).reshape([n_ell, 2,
                                                                n_ell, 2])

    covar_02_22 = nmt.gaussian_covariance(cw02_22, 0, 2, 2, 2,  # Spins of the 4 fields
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          w02, wb=w22).reshape([n_ell, 2,
                                                                n_ell, 4])

    covar_22_22 = nmt.gaussian_covariance(cw22_22, 2, 2, 2, 2,  # Spins of the 4 fields
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          w22, wb=w22).reshape([n_ell, 4,
                                                                n_ell, 4])

    fname = prefix_out + '_covTh.npz'
    np.savez_compressed(fname, cw0000=covar_00_00, cw00_02=covar_00_02,
                        cw00_22=covar_00_22, cw02_02=covar_02_02,
                        cw02_22=covar_02_22, cw22_22=covar_22_22)

#Generate theory prediction
if not os.path.isfile(prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(np.array([cltt])))
    cl02_th=w02.decouple_cell(w02.couple_cell(np.array([clte,clte])))
    cl22_th=w22.decouple_cell(w22.couple_cell(np.array([clee,clbe,cleb,clbb])))
    np.savetxt(o.prefix_out+"_cl_th.txt",
               np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                             cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))


#Compute mean and variance over nsims simulations
cl00_all=[]
cl02_all=[]
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)) :
        f0,f2=get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0))
        cl02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2))
        cl22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2))
        np.savez(o.prefix_out+"_cl_%04d"%(o.isim_ini+i),
                 l=b.get_effective_ells(),cltt=cl00[0],clte=cl02[0],cltb=cl02[1],
                 clee=cl22[0],cleb=cl22[1],clbe=cl22[2],clbb=cl22[3])
    cld=np.load(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cltt']])
    cl02_all.append([cld['clte'],cld['cltb']])
    cl22_all.append([cld['clee'],cld['cleb'],cld['clbe'],cld['clbb']])
cl00_all=np.array(cl00_all)
cl02_all=np.array(cl02_all)
cl22_all=np.array(cl22_all)

#Save output
np.savez(o.prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=b.get_effective_ells(),cl00=cl00_all,cl02=cl02_all,cl22=cl22_all)
