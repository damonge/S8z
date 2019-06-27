#!/usr/bin/python
import numpy as np
import os

def get_tracer_name(ibin):
    if ibin in np.arange(5):
        name = 'DESgc{}'.format(ibin)
    elif ibin in np.arange(5, 9):
        name = 'DESwl{}'.format(ibin-5)
    elif ibin == 9:
        name = 'PLAcv'

    return name

def split_cls_all_array(cls_all, lbpw, bins, index_B, outdir):
    nmaps = cls_all.shape[0]

    for i in range(nmaps):
        if i in index_B:
            continue
        for j in range(i, nmaps):
            if j in index_B:
                continue
            cl_bins = [bins[i], bins[j]]
            tracer_names = [get_tracer_name(ibin) for ibin in cl_bins]
            fname = os.path.join(outdir, 'cls_{}_{}.npz'.format(*tracer_names))
            np.savez_compressed(fname, ells=lbpw, cls=cls_all[i, j])

def correct_minus_sign_sh(cls_all, index_sh):
    for i in range(cls_all.shape[0]):
        for j in range(cls_all.shape[1]):
            f = 1
            if i in index_sh:
                f *= -1
            if j in index_sh:
                f *= -1
            cls_all[i, j] *= f

    return cls_all

if __name__ == '__main__':
    output_folder = '/mnt/extraspace/gravityls_3/S8z/Cls/all_together'

    # ####
    # #### Unnecessary anymore. It is done in the main script.
    # ####
    # index_sh = [5, 6, 7, 8, 9, 10, 11, 12]
    # fname = os.path.join(output_folder,  "cl_all_with_noise.npz")
    # cls_all_file = np.load(fname)
    # lbpw, cls_all = cls_all_file['l'], cls_all_file['cls']
    # cls_all = correct_minus_sign_sh(cls_all, index_sh)
    # np.savez_compressed(fname, l=lbpw, cls=cls_all)

    # fname = os.path.join(output_folder,  "cl_all_no_noise.npz")
    # cls_all_file = np.load(fname)
    # lbpw, cls_all = cls_all_file['l'], cls_all_file['cls']
    # cls_all = correct_minus_sign_sh(cls_all, index_sh)
    # np.savez_compressed(fname, l=lbpw, cls=cls_all)

    bins = [0, 1, 2, 3, 4] + [5, 5] + [6, 6] + [7, 7] + [8, 8] + [9]
    index_B = [6, 8, 10, 12]
    split_cls_all_array(cls_all, lbpw, bins, index_B, output_folder)
