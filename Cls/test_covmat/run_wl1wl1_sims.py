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
parser.add_option('--outdir',dest='outdir',default='./sims_wl1wl1',type=str,
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
prefix_out = os.path.join(o.outdir, 'run_wl1wl1')
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
# Define binning: needed to recompute workspaces
# The ells_lim_bpw
ells = np.arange(3 * nside)
ells_lim_bpw= np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])
ells_lim_bpw = ells_lim_bpw[ells_lim_bpw <= ells[-1] + 1]
bpws = np.zeros(ells.shape)
weights = np.zeros(ells.shape)

li = 0
for i, lf in enumerate(ells_lim_bpw[1:]):
    # lf += 1
    bpws[li : lf] = i
    weights[li : lf] += 1./weights[li : lf].size
    li = lf

b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)

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
# BB
clbb = np.zeros(clee.size)
# TB
cleb = clbe = cltb = 0 * clte

# Read noise
# gc3
fname = os.path.join(obs_path, 'des_w_cl_shot_noise_ns{}.npz'.format(nside))
nls = np.load(fname)
nltt = nls['cls'][3]
nltt = interp1d(nls['l'],  nltt, bounds_error=False,
                fill_value=(nltt[0], nltt[-1]))(l)
# wl1
fname = os.path.join(obs_path, 'des_sh_metacal_rot0-10_noise_ns{}.npz'.format(nside))
nls = np.load(fname)
nlee = nls['cls'][1, 0, 0]
nlbb = nls['cls'][1, 1, 1]
nlee = interp1d(nls['l'],  nlee, bounds_error=False,
                fill_value=(nlee[0], nlee[-1]))(l)
nlbb = interp1d(nls['l'],  nlbb, bounds_error=False,
                fill_value=(nlbb[0], nlbb[-1]))(l)
# gc3-wl1
nlte = np.zeros(l.size)

# These lines come from the PCLCovariance's run_sph_sims.py script
# cltt[0]=0
# nltt[0]=0

#Read mask
# fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
# mask_gc = hp.read_map(fname, verbose=False)

fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/map_metacal_bin1_counts_w_ns{}.fits'.format(nside)
mask_wl = hp.read_map(fname, verbose=False)

#Read bpw
fname = os.path.join(obs_path, 'l_bpw.txt')
lbpw = np.loadtxt(fname)
if not np.all(nls['l'] == lbpw):
    raise ValueError("lbpw != nls['l']")

# print(b.get_effective_ells())
# print(lbpw)
# print(cltt.shape)
# print(clte.shape)
# print(cltb.shape)
# print(clee.shape)
# print(cleb.shape)
# print(clbb.shape)

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
    # ff0=nmt.NmtField(mask_gc,[st], n_iter=0)
    ff2=nmt.NmtField(mask_wl,[sq, su], n_iter=0)
    return ff2

np.random.seed(1000)
f2 = get_fields()

#Compute mode-coupling matrix
w22 = nmt.NmtWorkspace();
if not os.path.isfile(prefix_out+"_w22_02.dat") : #spin0-spin2
    print("Computing w22")
    w22.compute_coupling_matrix(f2,f2,b, n_iter=0)
    w22.write_to(prefix_out+"_w22_02.dat");
else:
    w22.read_from(prefix_out+"_w22_02.dat")

# Compute covariance
if not os.path.isfile(prefix_out + '_covTh.npz'):
    cw = nmt.NmtCovarianceWorkspace()
    # This is the time-consuming operation
    # Note that you only need to do this once,
    # regardless of spin
    if not os.path.isfile(prefix_out+"_cw2222.dat") : #spin0-spin2
        print("Computing cw2222")
        cw.compute_coupling_coefficients(f2, f2, f2, f2, n_iter=0)
        cw.write_to(prefix_out + "_cw2222.dat")
    else:
        cw.read_from(prefix_out + '_cw2222.dat')

    cla1b1 = cla2b2 = cla1b2 = cla2b1 = [(clee + nlee), cleb, clbe, clbb + nlbb]

    s_a1 = s_b1 = 2
    s_a2 = s_b2 = 2
    ###
    wa = wb = w22


    C = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                                cla1b1, cla1b2, cla2b1, cla2b2,
                                wa, wb)
    fname = prefix_out + '_covTh.npz'
    np.savez_compressed(fname, C)

#Generate theory prediction
if not os.path.isfile(prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w22.decouple_cell(w22.couple_cell(np.array([clee, cleb, clbe, clbb])))
    np.savetxt(prefix_out+"_cl_th.txt",
               np.transpose([lbpw,cl00_th[0]]))


#Compute mean and variance over nsims simulations
cl00_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)) :
        f2 = get_fields()
        cl00=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2))#,cl_bias=clb00)
        np.savez(prefix_out+"_cl_%04d"%(o.isim_ini+i),
                 l=lbpw,cls=cl00[0])
    cld=np.load(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cls']])
cl00_all=np.array(cl00_all)

#Save output
np.savez(prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=lbpw,cls=cl00_all)
