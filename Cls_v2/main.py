#!/usr/bin/python

import os
import time
import common as co

##############################################################################
def launch_cls(data, queue, njobs):
    #######
    nc = 4
    mem = 5
    #
    cl_tracers = co.get_cl_tracers(data)
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
    nc = 4
    mem = 5
    #
    cov_tracers = co.get_cov_tracers(data)
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

    data = co.read_data(args.INPUT)

    queue = args.queue
    njobs = args.njobs

    if data['compute']['cls']:
        njobs = launch_cls(data, queue, njobs)
        # Remaining njobs to launch

    if data['compute']['cov'] and not args.no_cov:
        njobs = launch_cov(data, queue, njobs)
