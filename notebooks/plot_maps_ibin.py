#!/usr/bin/python3
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import healpy as hp
import matplotlib
import numpy as np
import os
os.environ['PATH'] += os.pathsep + '/mnt/zfsusers/gravityls_3/codes/latex/texlive/2020/bin/x86_64-linux'

#### Selection of Andrina's font config #####
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['text.usetex'] = True
####

CMAP = ListedColormap(np.loadtxt("Planck_Parchment_RGB.txt")/255.)
CMAP.set_bad("lightgray")
###############################################################################

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

def plot_map(imap, mask, fname, lims=True, unseen=True, cb=True, clabel=''):
    if lims is True:
        imin, imax = get_min_max(imap, mask, True)
    elif np.array(lims).size == 2:
        imin, imax = lims
    else:
        imin = imax = None

    if unseen:
        imap = imap.copy()
        imap[mask <= 0] = np.nan # hp.UNSEEN

    xsize = 2000
    ysize = int(xsize/2)

    lon = np.linspace(-70, 110, xsize)
    lat = np.linspace(-65, -35, ysize)
    LON, LAT = np.meshgrid(lon, lat)
    grid_pix = hp.ang2pix(nside, LON, LAT, lonlat=True)
    grid_map = imap[grid_pix]

    if cb:
        figsize = (8, 2.5)
    else:
        figsize = (8, 2)

    if imin == 0:
        cmap = 'viridis'
    else:
        cmap = CMAP

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    image = ax.pcolormesh(LON, LAT, grid_map[:, ::-1],
                           vmin=imin, vmax=imax,
                           rasterized=True, cmap=cmap)

    if cb:
        cbar = f.colorbar(image, orientation='horizontal',
                          shrink=.6, pad=0.21, ticks=[imin, imax])
        cbar.set_label(clabel, labelpad=-2)

        # remove white space around figure
        spacing = 0.1
        plt.subplots_adjust(bottom=spacing, top=1-spacing/2,
                            left=spacing/1.2, right=1-spacing/5)
    else:
        plt.tight_layout()

    # Modify ticklabels after resizing
    xtlabels = ['${:.0f}^\circ$'.format(i) for i in ax.get_xticks()]
    ytlabels = ['${:.0f}^\circ$'.format(i) for i in ax.get_yticks()]

    ax.set_xticklabels(xtlabels)
    ax.set_yticklabels(ytlabels)

    ax.set_xlabel(r'$\mathrm{RA}$')
    ax.set_ylabel(r'$\mathrm{Dec}$')


    plt.grid(True)
    f.savefig(outdir + fname)
    plt.close()

######
# File names
fn_psfE1 = 'map_{}_bin{}_wpsfe1_ns{}.fits'.format(wltype, ibin, nside)
fn_psfE2 = 'map_{}_bin{}_wpsfe2_ns{}.fits'.format(wltype, ibin, nside)
fn_mask = 'map_{}_bin{}_w_ns{}.fits'.format(wltype, ibin, nside)
fn_we1 = 'map_{}_bin{}_we1_ns{}.fits'.format(wltype, ibin, nside)
fn_we2 = 'map_{}_bin{}_we2_ns{}.fits'.format(wltype, ibin, nside)
map_list = [fn_psfE1, fn_psfE2, fn_we1, fn_we2]
label_list = [r'$e_{{\rm PSF}, 1}$', r'$e_{{\rm PSF}, 2}$', r'$e_1$', r'$e_2$']

# Load and plot mask
mask = hp.read_map(data_path + fn_mask, verbose=False)
plot_map(mask, mask, fn_mask.replace('.fits', '.pdf'), lims=True, unseen=False, clabel=r'$w$') #(1, 5))

for i, fname in enumerate(map_list):
    imap = hp.read_map(data_path + fname, verbose=False)

    ### Maps convolved with mask
    # plot_map(imap, mask, fname=fname.replace('.fits', '.pdf'), lims=True)

    # plt.hist(imap[mask>0], bins=60, density=True)
    # plt.savefig(outdir + fname.replace('.fits', '-hist.pdf'))
    # plt.close()

    ### Deconvolve (i.e. divide) the mask
    imap[mask > 0] /= mask[mask > 0]

    # if i%2:
    #     cb = True
    # else:
    #     cb = False
    cb = True

    plot_map(imap, mask, fname=fname.replace('.fits', '.pdf').replace('_w', '_'),
             lims=True, cb=cb, clabel=label_list[i])
    # plt.hist(imap[mask>0], bins=60, density=True)
    # plt.savefig(outdir + fname.replace('.fits', '-hist.pdf').replace('_w', '_'))
    # plt.close()
