#!/usr/bin/python

import os
import yaml

##############################################################################
def get_cl_tracers(data):
    cl_tracers = []
    tr_names = [trn for trn in data['tracers']]
    for i, tr1 in enumerate(tr_names):
        for tr2 in tr_names[i:]:
            trreq = ''.join(s for s in (tr1 + '-' + tr2) if not s.isdigit())
            clreq =  data['cls'][trreq]
            if clreq == 'all':
                pass
            elif (clreq == 'auto') and (tr1 != tr2):
                continue
            elif clreq is None:
                continue
            cl_tracers.append((tr1, tr2))

    return cl_tracers

def get_cov_tracers(data):
    cl_tracers = get_cl_tracers(data)
    cov_tracers = []
    for i, trs1 in enumerate(cl_tracers):
        for trs2 in cl_tracers:
            cov_tracers.append((*trs1, *trs2))

    return cov_tracers


def launch_cls(data):
    #######
    nc = 4
    queue = 'cmb'
    mem = 5
    #
    cl_tracers = get_cl_tracers(data)
    for tr1, tr2 in cl_tracers:
        comment = 'Cl({}-{})'.format(tr1, tr2)
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cls.py {} {} {}'.format(args.INPUT, tr1, tr2)
        # os.system(pyexec + " " + pyrun)
        print(pyexec + " " + pyrun)

def launch_cov(data):
    #######
    nc = 10
    queue = 'cmb'
    mem = 5
    #
    cov_tracers = get_cov_tracers(data)
    for trs in cov_tracers:
        comment = 'Cov({}-{},{}-{})'.format(*trs)
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cov.py {} {} {} {}'.format(args.INPUT, *trs)
        # os.system(pyexec + " " + pyrun)
        print(pyexec + " " + pyrun)


##############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    args = parser.parse_args()

    ##############################################################################

    with open(args.INPUT) as f:
        data = yaml.safe_load(f)

    if data['compute']['cls']:
        launch_cls(data)

    if data['compute']['cov']:
        launch_cov(data)
