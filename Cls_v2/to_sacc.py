#!/usr/bin/python
from cl import Cl
from cov import Cov
import common as co
import numpy as np
import sacc
import os

# TODO: move this to data.ylm?

class sfile():
    def __init__(self, datafile, output):
        self.datafile = datafile
        self.data = co.read_data(datafile)
        self.outdir = self.data['output']
        self.s = sacc.Sacc()
        self.add_tracers()
        self.add_ell_cls()
        self.add_covariance()
        fname = os.path.join(self.outdir, output)
        self.s.save_fits(fname, overwrite=True)

    def add_tracers(self):
        tracers = co.get_tracers_used(self.data)
        print(tracers)
        for tr in tracers:
            self.add_tracer(tr)

    def add_ell_cls(self):
        cl_tracers = co.get_cl_tracers(self.data)
        for tr1, tr2 in cl_tracers:
            self.add_ell_cl(tr1, tr2)

    def add_covariance(self):
        cov_tracers = co.get_cov_tracers(self.data)
        ndim = self.s.mean.size
        print(ndim)

        i_ini = i_end = j_ini = j_end = 0
        covmat = -1 * np.ones((ndim, ndim))
        for trs in cov_tracers:
            print(trs)
            if j_end >= ndim:
                j_ini = j_end = 0
            cov = Cov(self.datafile, *trs).cov
            ni, nj = cov.shape
            i_end += ni
            j_end += nj
            print(i_ini, i_end, j_ini, j_end)
            covmat[i_ini : i_end, j_ini : j_end] = cov
            covmat[j_ini : j_end, i_ini : i_end] = cov.T
            i_ini = i_end
            j_ini = j_end

        self.s.add_covariance(covmat)

    def add_tracer(self, tr):
        tracer = self.data['tracers'][tr]
        dndz = np.loadtxt(tracer['dndz'])
        z=dndz[1]
        nz=dndz[3]

        if tracer['type'] == 'gc':
            quantity = 'galaxy_density'
        elif tracer['type'] == 'wl':
            quantity = 'galaxy_shear'
        elif tracer['type'] == 'cv':
            quantity = 'cmb_convergence'
        else:
            raise ValueError('Tracer type {} not implemented'.format(tracer['type']))

        self.s.add_tracer('NZ', tr, quantity=quantity, spin=tracer['spin'],
                          z=z, nz=nz)

    def add_ell_cl(self, tr1, tr2, nl=False):
        ells_nobin = np.arange(3 * self.data['healpy']['nside'])
        cl = Cl(self.datafile, tr1, tr2)
        w = cl.get_workspace()
        ws_bpw= w.get_bandpower_windows()

        wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[0, :, 0, :].T)

        if cl.cl.shape[0] == 1:
            cl_types = ['cl_00']
        elif cl.cl.shape[0] == 2:
            cl_types = ['cl_0e', 'cl_0b']
        else:
            cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']


        for i, cl_type in enumerate(cl_types):
            if nl:
                cli = cl.nl[i]
            else:
                cli = cl.cl[i]
            self.s.add_ell_cl(cl_type, tr1, tr2, cl.ell, cli, window=wins)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('output', type=str, help="Name of the generated sacc file. Stored in yml['output']")
    args = parser.parse_args()

    sfile = sfile(args.INPUT, args.output)
