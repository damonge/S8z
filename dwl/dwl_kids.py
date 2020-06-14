import wget
import os
from dwl_utils import dwl_file, mkdir, unzip, untar_and_remove

predir=os.path.abspath('/mnt/extraspace/damonge/S8z_data')

# Create data directory
mkdir(predir+"/KiDS_data")
os.chdir(predir+"/KiDS_data")

# Download all files
mkdir("shear_KV450_catalog")
os.chdir("shear_KV450_catalog")
# Download summary file
dwl_file("KV450_G9_reweight_3x4x4_v2_good.cat", "http://ds.astro.rug.astro-wise.org:8000/KV450_G9_reweight_3x4x4_v2_good.cat")
dwl_file("KV450_G12_reweight_3x4x4_v2_good.cat", "http://ds.astro.rug.astro-wise.org:8000/KV450_G12_reweight_3x4x4_v2_good.cat")
dwl_file("KV450_G15_reweight_3x4x4_v2_good.cat", "http://ds.astro.rug.astro-wise.org:8000/KV450_G15_reweight_3x4x4_v2_good.cat")
dwl_file("KV450_G23_reweight_3x4x4_v2_good.cat", "http://ds.astro.rug.astro-wise.org:8000/KV450_G23_reweight_3x4x4_v2_good.cat")
dwl_file("KV450_GS_reweight_3x4x4_v2_good.cat", "http://ds.astro.rug.astro-wise.org:8000/KV450_GS_reweight_3x4x4_v2_good.cat")
os.chdir("../")

# Data vectors and redshift distributions
dwl_file("KV450_COSMIC_SHEAR_DATA_RELEASE/SUPPLEMENTARY_FILES/CUT_VALUES/cut_values_5zbins_zbin5_small_scales.txt",
         "http://kids.strw.leidenuniv.nl/cs2018/KV450_COSMIC_SHEAR_DATA_RELEASE.tar.gz",
         call=lambda : untar_and_remove("KV450_COSMIC_SHEAR_DATA_RELEASE.tar.gz"))

# Go back to root
os.chdir(predir)
