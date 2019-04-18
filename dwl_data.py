import wget
import os

predir=os.path.abspath('./data')

def dwl_file(fname,url,call=None,verbose=True):
    if not os.path.isfile(fname):
        if verbose:
            print(fname)
        wget.download(url)
        print("\n")
        if call is not None:
            call()

def mkdir(dr):
    if not os.path.isdir(dr):
        os.makedirs(dr)

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
    dwl_file(fname,l)
os.chdir("../")
f.close()

mkdir("syst_maps")
os.chdir("syst_maps")
# Download footprint
dwl_file("y1a1_gold_1.0.2_wide_footprint_4096.fits.gz","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/masks_maps/y1a1_gold_1.0.2_wide_footprint_4096.fits.gz")
dwl_file("y1a1_gold_1.0.3_wide_badmask_4096.fits.gz","http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/masks_maps/y1a1_gold_1.0.3_wide_badmask_4096.fits.gz")

#Observing conditions
def unzip(fname):
    os.system('unzip '+fname)
    os.remove(fname)
for filt in ['g','r','i','z','Y']:
    dwl_file("Y1A1GOLD_band_"+filt+"_nside4096_count__fracdet.fits.gz",
             "http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/systematic_maps/Y1A1GOLD_band_"+filt+"_nside4096_systematics.zip",
             lambda : unzip("Y1A1GOLD_band_"+filt+"_nside4096_systematics.zip"))
os.chdir("../")

# Go back to root
os.chdir(predir)
