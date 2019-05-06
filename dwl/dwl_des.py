import os
from dwl_utils import dwl_file, mkdir, unzip

predir=os.path.abspath('/mnt/bluewhale/damonge/S8z_data')

# Create data directory
mkdir(predir+"/DES_data")
os.chdir(predir+"/DES_data")

# Download summary file
dwl_file("ALL_FILES.txt","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/ALL_FILES.txt")

# Download all files
f=open("ALL_FILES.txt","r")
mkdir("gold_catalog")
os.chdir("gold_catalog")
for l in f:
    l=l.rstrip('\n')
    fname=l.split('/')[-1]
#    dwl_file(fname,l)
os.chdir("../")
f.close()

mkdir("syst_maps")
os.chdir("syst_maps")
# Download footprint
dwl_file("y1a1_gold_1.0.2_wide_footprint_4096.fits.gz","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/masks_maps/y1a1_gold_1.0.2_wide_footprint_4096.fits.gz")
dwl_file("y1a1_gold_1.0.3_wide_badmask_4096.fits.gz","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/masks_maps/y1a1_gold_1.0.3_wide_badmask_4096.fits.gz")

# Observing conditions and depth
for filt in ['g','r','i','z','Y']:
    dwl_file("y1a1_gold_1.0.2_wide_auto_nside4096_"+filt+"_10sigma.fits.gz",
             "http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/masks_maps/y1a1_gold_1.0.2_wide_auto_nside4096_"+filt+"_10sigma.fits.gz")
    dwl_file("Y1A1GOLD_band_"+filt+"_nside4096_count__fracdet.fits.gz",
             "http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/systematic_maps/Y1A1GOLD_band_"+filt+"_nside4096_systematics.zip",
             lambda : unzip("Y1A1GOLD_band_"+filt+"_nside4096_systematics.zip"))
os.chdir("../")

# Data vector
mkdir("data_vector")
os.chdir("data_vector")
dwl_file("2pt_NG_mcal_1110.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits")
os.chdir("../")

# Clustering sample
mkdir("redmagic_catalog")
os.chdir("redmagic_catalog")
dwl_file("DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redmagic/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits")
dwl_file("DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redmagic/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits")
dwl_file("DES_Y1A1_3x2pt_redMaGiC_RANDOMS.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redmagic/DES_Y1A1_3x2pt_redMaGiC_RANDOMS.fits")
os.chdir("../")

# Shear sample
mkdir("shear_catalog")
os.chdir("shear_catalog")
dwl_file("mcal-y1a1-combined-riz-unblind-v4-matched.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/shear_catalogs/mcal-y1a1-combined-riz-unblind-v4-matched.fits")
dwl_file("y1a1-im3shape_v5_unblind_v2_matched_v4.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/shear_catalogs/y1a1-im3shape_v5_unblind_v2_matched_v4.fits")
dwl_file("y1_source_redshift_binning_v1.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redshift_bins/y1_source_redshift_binning_v1.fits")
dwl_file("y1_redshift_distributions_v1.fits","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redshift_bins/y1_redshift_distributions_v1.fits")
os.chdir("../")

# Go back to root
os.chdir(predir)
