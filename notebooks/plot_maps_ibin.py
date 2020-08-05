#!/usr/bin/python3
from matplotlib import pyplot as plt
import argparse
import healpy as hp
import numpy as np


parser = argparse.ArgumentParser(description="Plot maps for shear paper")
parser.add_argument('--nside',  default=4096, type=int,
                    help='HEALPix nside param')
parser.add_argument('--wltype', default='metacal', type=str,
                    help='DES weak lensing shear measurement algorithm (metacal or im3shape)')
parser.add_argument('--ibin',  default=1, type=int,
                    help='DES weak lensing z-bin')
parser.add_argument('--outdir', default='./figures_shear/', type=str,
                  help='Path to save output')
parser.add_argument('--plot', default=False, action='store_true',
                    help='Set if you want to produce plots')

args = parser.parse_args()
###############################################################################

wltype = args.wltype
nside = args.nside
ibin = args.ibin
outdir = args.outdir

data_path = '/mnt/extraspace/damonge/S8z_data/derived_products/des_shear/'
##############################################################################
def get_min_max(imap, mask, iround=False):
    if np.all(imap == mask):
        imin = np.min(mask)
        mask = mask[mask > 0].copy()
        mask.sort()
        ix = round(0.99*mask.size)
        if ix < mask.size:
            imax = mask[ix]
        else:
            imax = mask[-1]
        return imin, imax
    m = imap[mask > 0]
    mean = np.mean(m)
    sigma = np.sqrt(np.cov(m))
    imin = mean - 3*sigma
    imax = mean + 3*sigma
    if iround:
        if np.abs(imin) < 1:
            om = -int(np.floor(np.log10(np.abs(imin))))
        else:
            om = 1
        imin = round(imin, om)
        imax = round(imax, om)
    return imin, imax

def plot_map(imap, mask, fname, lims=True, unseen=True, cb=True):
    f = plt.figure(1, figsize=(8, 2)) #, tight_layout=True)

    if lims is True:
        imin, imax = get_min_max(imap, mask, True)
    elif np.array(lims).size == 2:
        imin, imax = lims
    else:
        imin = imax = None

    if unseen:
        imap = imap.copy()
        imap[mask <= 0] = hp.UNSEEN

    hp.cartview(imap, fig=1, lonra=[-70, 110], latra=[-65, -35], title='',
                margins=(0.01, 0.1, 0.01, 0.005),
                min=imin, max=imax, cbar=cb)
    plt.savefig(outdir + fname)
    plt.close()

######
# File names
fn_psfE1 = 'map_{}_bin{}_wpsfe1_ns{}.fits'.format(wltype, ibin, nside)
fn_psfE2 = 'map_{}_bin{}_wpsfe2_ns{}.fits'.format(wltype, ibin, nside)
fn_mask = 'map_{}_bin{}_w_ns{}.fits'.format(wltype, ibin, nside)
fn_we1 = 'map_{}_bin{}_we1_ns{}.fits'.format(wltype, ibin, nside)
fn_we2 = 'map_{}_bin{}_we2_ns{}.fits'.format(wltype, ibin, nside)
map_list = [fn_psfE1, fn_psfE2, fn_we1, fn_we2]

# Load and plot mask
mask = hp.read_map(data_path + fn_mask, verbose=False)
plot_map(mask, mask, fn_mask.replace('.fits', '.pdf'), lims=True, unseen=False) #(1, 5))

for i, fname in enumerate(map_list):
    imap = hp.read_map(data_path + fname, verbose=False)

    ### Maps convolved with mask
    # plot_map(imap, mask, fname=fname.replace('.fits', '.pdf'), lims=True)

    # plt.hist(imap[mask>0], bins=60, density=True)
    # plt.savefig(outdir + fname.replace('.fits', '-hist.pdf'))
    # plt.close()

    ### Deconvolve (i.e. divide) the mask
    imap[mask > 0] /= mask[mask > 0]

    if i%2:
        cb = True
    else:
        cb = False

    plot_map(imap, mask, fname=fname.replace('.fits', '.pdf').replace('_w', '_'),
             lims=True, cb=cb)
    # plt.hist(imap[mask>0], bins=60, density=True)
    # plt.savefig(outdir + fname.replace('.fits', '-hist.pdf').replace('_w', '_'))
    # plt.close()
