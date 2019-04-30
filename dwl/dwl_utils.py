import wget
import os

def untar_and_remove(fname):
    os.system("tar -xf " + fname)
    os.remove(fname)

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

def unzip(fname):
    os.system('unzip '+fname)
    os.remove(fname)
