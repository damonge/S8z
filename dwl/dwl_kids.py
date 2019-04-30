import wget
import os
from dwl_utils import dwl_file, mkdir, unzip, untar_and_remove

predir=os.path.abspath('/bluewhale/damonge/S8z_data')

# Create data directory
mkdir(predir+"/KiDS_data")
os.chdir(predir+"/KiDS_data")

# Download all files
mkdir("shear_dr3_catalog")
os.chdir("shear_dr3_catalog")
# Download summary file
dwl_file("kids_dr3.1_shear_wget.txt","http://kids.strw.leidenuniv.nl/DR3/kids_dr3.1_shear_wget.txt")
# Download individual files
f=open("kids_dr3.1_shear_wget.txt","r")
for l in f:
    l=l.rstrip('\n')
    fname=l.split('/')[-1]
    dwl_file(fname,l)
f.close()
os.chdir("../")

# Download footprint
mkdir("footprint")
os.chdir("footprint")
dwl_file("kids-450_footprint_mask.fits.gz","http://kids.strw.leidenuniv.nl/DR3/kids-450_footprint_mask.fits.gz")
os.chdir('../')

# Data vectors and redshift distributions
dwl_file("KiDS-450_COSMIC_SHEAR_DATA_RELEASE/Nz_CC/Nz_CC_z0.1t0.3.asc",
         "http://kids.strw.leidenuniv.nl/cs2016/KiDS-450_COSMIC_SHEAR_DATA_RELEASE.tar.gz",
         call=lambda : untar_and_remove("KiDS-450_COSMIC_SHEAR_DATA_RELEASE.tar.gz"))

# Go back to root
os.chdir(predir)
