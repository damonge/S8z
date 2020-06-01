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
parser.add_option('--outdir',dest='outdir',default='./sims_gc3_wl1_all',type=str,
                  help='Output directory')
parser.add_option('--nside', dest='nside', default=512, type=int,
                  help='HEALPix nside param')
parser.add_option('--n_iter', dest='n_iter', default=3, type=int,
                  help='n_iter for compute_coupling_matrix and compute_coupling_coefficients')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')

(o, args) = parser.parse_args()

##############################################################################
# Set files prefix
prefix_out = os.path.join(o.outdir, 'run_gc3_wl1_all')
# Set nside
valid_nside = [512, 2048, 4096]
# wltype
wltype = 'im3shape'
# wltype = 'metacal'  # not for 512 with newbin
# Set root path of observations
obs_path = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together'
if o.nside != 4096:
    obs_path += '_{}_{}_newbin'.format(wltype, o.nside)
##############################################################################
# Check obspath:
if not os.path.exists(obs_path):
    raise ValueError('Obs path does not exist: {}'.format(obs_path))
# Check nside
if not o.nside in valid_nside:
    raise ValueError('nside must be one of {}'.format(valid_nside))
nside = o.nside  # 2048
# Create outdir
os.makedirs(o.outdir, exist_ok=True)
# Set nsims
nsims=o.isim_end-o.isim_ini+1

# Define binning: needed to recompute workspaces
# The ells_lim_bpw
# The ells_lim_bpw
ells = np.arange(3 * nside)
ells_lim_bpw= np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])
ells_lim_bpw = ells_lim_bpw[ells_lim_bpw <= 3 * nside] # 3*nside == ells[-1] + 1
if 3*nside not in ells_lim_bpw: # Exhaust lmax --> gives same result as previous method, but adds 1 bpw (not for 4096)
    ells_lim_bpw = np.append(ells_lim_bpw, 3*nside)
b = nmt.NmtBin.from_edges(ells_lim_bpw[:-1], ells_lim_bpw[1:])

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
# gc3
fname = os.path.join(obs_path, 'des_w_cl_shot_noise_ns{}.npz'.format(nside))
nls = np.load(fname)
nltt = nls['cls'][3]
nltt = interp1d(nls['l'],  nltt, bounds_error=False,
                fill_value=(nltt[0], nltt[-1]))(l)
# wl1
fname = os.path.join(obs_path, 'des_sh_{}_rot0-10_noise_ns{}.npz'.format(wltype, nside)) 
nls = np.load(fname)
nlee = nls['cls'][1, 0, 0]
nlbb = nls['cls'][1, 1, 1]
nlee = interp1d(nls['l'],  nlee, bounds_error=False,
                fill_value=(nlee[0], nlee[-1]))(l)
nlbb = interp1d(nls['l'],  nlbb, bounds_error=False,
                fill_value=(nlbb[0], nlbb[-1]))(l)
# gc3-wl1
nlte = np.zeros(l.size)

#Read mask
fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
mask_gc = hp.read_map(fname, verbose=False)

fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/map_{}_bin1_counts_w_ns{}.fits'.format(wltype, nside)
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
    st,sq,su=hp.synfast([cltt+nltt,clee+nlee,clbb+nlbb,clte+nlte],o.nside,new=True,verbose=False,pol=True)
    ff0=nmt.NmtField(mask_gc,[st],n_iter=o.n_iter)
    ff2=nmt.NmtField(mask_wl,[sq, su],n_iter=o.n_iter)
    return ff0, ff2

np.random.seed(1000)
f0, f2 = get_fields()

##############################################################################
#Compute mode-coupling matrix
##############################################################################
w00 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w00_00.dat") : #spin0-spin2
    print("Computing w00")
    w00.compute_coupling_matrix(f0,f0,b, n_iter=o.n_iter)
    w00.write_to(prefix_out+"_w00_00.dat");
else:
    w00.read_from(prefix_out+"_w00_00.dat")

w02 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w02_02.dat") : #spin0-spin2
    print("Computing w02")
    w02.compute_coupling_matrix(f0,f2,b, n_iter=o.n_iter)
    w02.write_to(prefix_out+"_w02_02.dat");
else:
    w02.read_from(prefix_out+"_w02_02.dat")

w22 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w22_22.dat") : #spin0-spin2
    print("Computing w22")
    w22.compute_coupling_matrix(f2,f2,b, n_iter=o.n_iter)
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
        cw00_00.compute_coupling_coefficients(f0, f0, f0, f0, n_iter=o.n_iter)
        cw00_00.write_to(prefix_out + "_cw0000.dat")
    else:
        cw00_00.read_from(prefix_out + '_cw0000.dat')

    cw00_02 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0002.dat") : # mask0-mask0-mask0-mask2
        print("Computing cw0002")
        cw00_02.compute_coupling_coefficients(f0, f0, f0, f2, n_iter=o.n_iter)
        cw00_02.write_to(prefix_out + "_cw0002.dat")
    else:
        cw00_02.read_from(prefix_out + '_cw0002.dat')

    cw00_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0022.dat") : # mask0-mask0-mask2-mask2
        print("Computing cw0022")
        cw00_22.compute_coupling_coefficients(f0, f0, f2, f2, n_iter=o.n_iter)
        cw00_22.write_to(prefix_out + "_cw0022.dat")
    else:
        cw00_22.read_from(prefix_out + '_cw0022.dat')

    cw02_02 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0202.dat") : # mask0-mask2-mask0-mask2
        print("Computing cw0202")
        cw02_02.compute_coupling_coefficients(f0, f2, f0, f2, n_iter=o.n_iter)
        cw02_02.write_to(prefix_out + "_cw0202.dat")
    else:
        cw02_02.read_from(prefix_out + '_cw0202.dat')

    cw02_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw0222.dat") : # mask0-mask2-mask2-mask2
        print("Computing cw0222")
        cw02_22.compute_coupling_coefficients(f0, f2, f2, f2, n_iter=o.n_iter)
        cw02_22.write_to(prefix_out + "_cw0222.dat")
    else:
        cw02_22.read_from(prefix_out + '_cw0222.dat')

    cw22_22 = nmt.NmtCovarianceWorkspace()
    if not os.path.isfile(prefix_out+"_cw2222.dat") : # mask2-mask2-mask2-mask2
        print("Computing cw2222")
        cw22_22.compute_coupling_coefficients(f2, f2, f2, f2, n_iter=o.n_iter)
        cw22_22.write_to(prefix_out + "_cw2222.dat")
    else:
        cw22_22.read_from(prefix_out + '_cw2222.dat')

    ##
    # Compute matrices
    ##
    covar_00_00 = nmt.gaussian_covariance(cw00_00,
                                          0, 0, 0, 0,  # Spins of the 4 fields
                                          [cltt+nltt],  # TT
                                          [cltt+nltt],  # TT
                                          [cltt+nltt],  # TT
                                          [cltt+nltt],  # TT
                                          w00, wb=w00).reshape([lbpw.size, 1,
                                                                lbpw.size, 1])

    covar_00_02 = nmt.gaussian_covariance(cw00_02, 0, 0, 0, 2,  # Spins of the 4 fields
                                          [cltt+nltt],  # TT
                                          [clte, cltb],  # TE, TB
                                          [cltt+nltt],  # TT
                                          [clte, cltb],  # TE, TB
                                          w00, wb=w02).reshape([lbpw.size, 1,
                                                                lbpw.size, 2])

    covar_00_22 = nmt.gaussian_covariance(cw00_22, 0, 0, 2, 2,  # Spins of the 4 fields
                                          [clte, cltb],  # TE, TB
                                          [clte, cltb],  # TE, TB
                                          [clte, cltb],  # TE, TB
                                          [clte, cltb],  # TE, TB
                                          w00, wb=w22).reshape([lbpw.size, 1,
                                                                lbpw.size, 4])

    covar_02_02 = nmt.gaussian_covariance(cw02_02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [cltt+nltt],  # TT
                                          [clte, cltb],  # TE, TB
                                          [clte, cltb],  # ET, BT
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          w02, wb=w02).reshape([lbpw.size, 2,
                                                                lbpw.size, 2])

    covar_02_22 = nmt.gaussian_covariance(cw02_22, 0, 2, 2, 2,  # Spins of the 4 fields
                                          [clte, cltb],  # TE, TB
                                          [clte, cltb],  # TE, TB
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          w02, wb=w22).reshape([lbpw.size, 2,
                                                                lbpw.size, 4])

    covar_22_22 = nmt.gaussian_covariance(cw22_22, 2, 2, 2, 2,  # Spins of the 4 fields
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          [clee+nlee, cleb,
                                           cleb, clbb+nlbb],  # EE, EB, BE, BB
                                          w22, wb=w22).reshape([lbpw.size, 4,
                                                                lbpw.size, 4])

    fname = prefix_out + '_covTh.npz'
    np.savez_compressed(fname, cw00_00=covar_00_00, cw00_02=covar_00_02,
                        cw00_22=covar_00_22, cw02_02=covar_02_02,
                        cw02_22=covar_02_22, cw22_22=covar_22_22)

#Generate theory prediction
if not os.path.isfile(prefix_out+'_clth.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(np.array([cltt])))
    cl02_th=w02.decouple_cell(w02.couple_cell(np.array([clte,cltb])))
    cl22_th=w22.decouple_cell(w22.couple_cell(np.array([clee,clbe,cleb,clbb])))
    np.savetxt(prefix_out+"_clth.txt",
               np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                             cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))


#Compute mean and variance over nsims simulations
cl00_all=[]
cl02_all=[]
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix_out+"_cl%04d.npz"%(o.isim_ini+i)) :
        f0,f2=get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0))
        cl02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2))
        cl22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2))
        np.savez(prefix_out+"_cl%04d"%(o.isim_ini+i),
                 l=b.get_effective_ells(),cltt=cl00[0],clte=cl02[0],cltb=cl02[1],
                 clee=cl22[0],cleb=cl22[1],clbe=cl22[2],clbb=cl22[3])
    cld=np.load(prefix_out+"_cl%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cltt']])
    cl02_all.append([cld['clte'],cld['cltb']])
    cl22_all.append([cld['clee'],cld['cleb'],cld['clbe'],cld['clbb']])
cl00_all=np.array(cl00_all)
cl02_all=np.array(cl02_all)
cl22_all=np.array(cl22_all)

#Save output
np.savez(prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=b.get_effective_ells(),cl00=cl00_all,cl02=cl02_all,cl22=cl22_all)
