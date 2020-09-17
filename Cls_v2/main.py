#!/usr/bin/python

import os
import time
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
            elif clreq == 'None':
                continue
            cl_tracers.append((tr1, tr2))

    return cl_tracers

def get_cov_tracers(data):
    cl_tracers = get_cl_tracers(data)
    cov_tracers = []
    for i, trs1 in enumerate(cl_tracers):
        for trs2 in cl_tracers[i:]:
            cov_tracers.append((*trs1, *trs2))

    return cov_tracers


def launch_cls(data, queue, njobs):
    #######
    nc = 4
    mem = 5
    #
    cl_tracers = get_cl_tracers(data)
    outdir = data['output']
    c = 0
    for tr1, tr2 in cl_tracers:
        if c >= njobs:
            break
        comment = 'cl_{}_{}'.format(tr1, tr2)
        # TODO: don't hard-code it!
        trreq = ''.join(s for s in (tr1 + '_' + tr2) if not s.isdigit())
        fname = os.path.join(outdir, trreq, comment + '.npz')
        if os.path.isfile(fname):
            continue

        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cl.py {} {} {}'.format(args.INPUT, tr1, tr2)
        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)

    return njobs - c

def launch_cov(data, queue, njobs):
    #######
    nc = 10
    mem = 5
    #
    cov_tracers = get_cov_tracers(data)
    outdir = data['output']
    c = 0
    for trs in cov_tracers:
        if c >= njobs:
            break
        comment = 'cov_{}_{}_{}_{}'.format(*trs)
        fname = os.path.join(outdir, 'cov', comment + '.npz')
        if os.path.isfile(fname):
            continue
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cov.py {} {} {} {} {}'.format(args.INPUT, *trs)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)

    return njobs - c


##############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('--queue', type=str, default='berg', help='SLURM queue to use')
    parser.add_argument('--no_cov', action='store_true', help='Do not compute covs even if set in the yml file')
    parser.add_argument('--njobs', type=int, default=20, help='Maximum number of jobs to launch')
    args = parser.parse_args()

    ##############################################################################

    with open(args.INPUT) as f:
        data = yaml.safe_load(f)

    queue = args.queue
    njobs = args.njobs

    if data['compute']['cls']:
        njobs = launch_cls(data, queue, njobs)
        # Remaining njobs to launch

    if data['compute']['cov'] and not args.no_cov:
        njobs = launch_cov(data, queue, njobs)
