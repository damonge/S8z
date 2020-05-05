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
parser.add_option('--outdir',dest='outdir',default='./sims',type=str,
                  help='Output directory')
# parser.add_option('--nside', dest='nside', default=512, type=int,
#                   help='HEALPix nside param')
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
prefix_out = os.path.join(o.outdir, 'run_gc0gc0')
# Set nside
nside = 2048
# Set nsims
nsims=o.isim_end-o.isim_ini+1
##############################################################################

##############################################################################
#Read input power spectra
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc0_DESgc0.npz'
gc0gc0 = np.load(fname)
l, cltt = gc0gc0['ells'], gc0gc0['cls']

fname = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_2048/des_w_cl_shot_noise_ns2048.npz'
nls = np.load(fname)
nltt = interp1d(nls['l'],  nls['cls'][0], bounds_error=False,
                fill_value=(nls['cls'][0, 0], nls['cls'][0, -1]))(l)
cltt=cltt[:3*nside]
nltt=nltt[:3*nside]
# These lines come from the PCLCovariance's run_sph_sims.py script
# cltt[0]=0
# nltt[0]=0

#Read mask
fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns2048.fits'
mask_lss = hp.read_map(fname, verbose=False)

#Read bpw
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together_2048/l_bpw.txt'
lbpw = np.loadtxt(fname)
if not np.all(nls['l'] == lbpw):
    raise ValueError("lbpw != nls['l']")
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
    st=hp.synfast(cltt+nltt,nside,new=True,verbose=False,pol=True)
    ff0=nmt.NmtField(mask_lss,[st])
    return ff0

np.random.seed(1000)
f0 = get_fields()

#Compute mode-coupling matrix
# TODO: Define outdir
w00=nmt.NmtWorkspace();
w00.read_from('/mnt/extraspace/gravityls_3/S8z/Cls/all_together_2048/w00_00.dat')

#Generate theory prediction
if not os.path.isfile(prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(np.array([cltt])))
    np.savetxt(prefix_out+"_cl_th.txt",
               np.transpose([lbpw,cl00_th[0]]))


#Compute mean and variance over nsims simulations
cl00_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)) :
        f0 = get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0))#,cl_bias=clb00)
        np.savez(prefix_out+"_cl_%04d"%(o.isim_ini+i),
                 l=lbpw,cltt=cl00[0])
    cld=np.load(prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cltt']])
cl00_all=np.array(cl00_all)

#Save output
np.savez(prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=lbpw,cl00=cl00_all)
