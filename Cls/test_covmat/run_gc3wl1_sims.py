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
parser.add_option('--outdir',dest='outdir',default='./sims_gc3wl1',type=str,
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
prefix_out = os.path.join(o.outdir, 'run_gc3wl1')
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
l, clte = fid_data['ells'], fid_data['cls']
# gc3 - gc3
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESgc3_DESgc3.npz'
cltt = np.load(fname)['cls']
# wl1 - wl1
fname = '/mnt/extraspace/gravityls_3/S8z/Cls/fiducial/nobaryons/cls_DESwl1_DESwl1.npz'
clee = np.load(fname)['cls']
# BB
clbb = np.zeros(clee.size)
# TB
cltb = 0 * clte

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

# Remove extra ells
cltt=cltt[:3*nside]
nltt=nltt[:3*nside]
clee=clee[:3*nside]
nlee=nlee[:3*nside]
clbb=clbb[:3*nside]
nlbb=nlbb[:3*nside]
# These lines come from the PCLCovariance's run_sph_sims.py script
# cltt[0]=0
# nltt[0]=0

#Read mask
fname = '/mnt/extraspace/damonge/S8z_data/derived_products/des_clustering/mask_ns{}.fits'.format(nside)
mask_lss = hp.read_map(fname, verbose=False)

#Read bpw
fname = os.path.join(obs_path, 'l_bpw.txt')
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
    st,sq,su=hp.synfast([cltt+nltt,clee+nlee,clbb+nlbb,clte+nlte],o.nside,new=True,verbose=False,pol=True)
    ff0=nmt.NmtField(mask_lss,[st])
    ff2=nmt.NmtField(mask_lss,[sq, su])
    return ff0, ff2

np.random.seed(1000)
f0, f2 = get_fields()

#Compute mode-coupling matrix
w02 = nmt.NmtWorkspace();
w02.read_from(os.path.join(obs_path, 'w02_02.dat'))

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
