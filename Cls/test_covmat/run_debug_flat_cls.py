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
clte = np.mean(clte) * np.ones(l.size)
cltt = np.mean(cltt) * np.ones(l.size)
clee = np.mean(clee) * np.ones(l.size)

#Read mask
fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
mask_lss = hp.read_map(fname, verbose=False)

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
    ff0=nmt.NmtField(mask_lss,[st])
    ff2=nmt.NmtField(mask_lss,[sq, su])
    return ff0, ff2

np.random.seed(1000)
f0, f2 = get_fields()

#Compute mode-coupling matrix
w02 = nmt.NmtWorkspace();
w02.read_from(os.path.join(obs_path, 'w02_02.dat'))

# Compute covariance
if not os.path.isfile(prefix_out + '_covTh.npz'):
    cla1b1 = (cltt + nltt).reshape(1, -1)
    cla2b2 = np.array([(clee + nlee), cleb, clbe, clbb + nlbb])
    cla1b2 = cla2b1 = np.array([clte, cltb])
    s_a1 = s_b1 = 0
    s_a2 = s_b2 = 2
    ###
    wa = wb = w02

    cw = nmt.NmtCovarianceWorkspace()
    cw.read_from(os.path.join(obs_path, 'cw0202.dat'))

    C = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                                cla1b1, cla1b2, cla2b1, cla2b2,
                                wa, wb)
    fname = prefix_out + '_covTh.npz'
    np.savez_compressed(fname, C)

#Generate theory prediction
if not os.path.isfile(prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w02.decouple_cell(w02.couple_cell(np.array([clte, cltb])))
    np.savetxt(prefix_out+"_cl_th.txt",
               np.transpose([lbpw,cl00_th[0]]))


#Compute mean and variance over nsims simulations
cl00_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)) :
        f0, f2 = get_fields()
        cl00=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2))#,cl_bias=clb00)
        np.savez(prefix_out+"_cl_%04d"%(o.isim_ini+i),
                 l=lbpw,cls=cl00[0])
    cld=np.load(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cls']])
cl00_all=np.array(cl00_all)

#Save output
np.savez(prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=lbpw,cls=cl00_all)
