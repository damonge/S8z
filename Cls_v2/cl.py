#!/usr/bin/python
import numpy as np
import healpy as hp
import pymaster as nmt
import os
import yaml

class Cl():
    def __init__(self, data, tr1, tr2):
        self.data = self.read_data(data)
        self.tr1 = tr1
        self.tr2 = tr2
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.b = self.get_NmtBin()
        self.ell, self.cl = self.get_ell_cl()

    def read_data(self, data):
        with open(data) as f:
            data = yaml.safe_load(f)
        return data

    def get_outdir(self):
        root = self.data['output']
        trreq = ''.join(s for s in (self.tr1 + '_' + self.tr2) if not s.isdigit())
        outdir = os.path.join(root, trreq)
        return outdir

    def get_NmtBin(self):
        bpw_edges = np.array(self.data['bpw_edges'])
        nside = self.data['healpy']['nside']
        bpw_edges = bpw_edges[bpw_edges <= 3 * nside] # 3*nside == ells[-1] + 1
        if 3*nside not in bpw_edges: # Exhaust lmax --> gives same result as previous method, but adds 1 bpw (not for 4096)
            bpw_edges = np.append(bpw_edges, 3*nside)
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b

    def get_field(self, tr):
        data = self.data
        tracers = data['tracers']
        # TODO: This can be optimize for the case mask1 == mask2
        mask = hp.read_map(tracers[tr]['mask'])
        maps = []
        if tracers[tr]['spin'] == 0:
            maps.append(hp.read_map(tracers[tr]['path']))
        else:
            maps.append(hp.read_map(tracers[tr]['path1']))
            maps.append(hp.read_map(tracers[tr]['path2']))
        f = nmt.NmtField(mask, maps, n_iter=data['healpy']['n_iter'])
        return f

    def load_fields(self):
        self.f1 = self.get_field(self.tr1)
        self.f2 = self.get_field(self.tr2)
        return self.f1, self.f2

    def get_workspace(self):
        mask1 = os.path.basename(self.data['tracers'][self.tr1]['mask'])
        mask2 = os.path.basename(self.data['tracers'][self.tr2]['mask'])
        # Remove the extension
        mask1 = os.path.splitext(mask1)[0]
        mask2 = os.path.splitext(mask2)[0]
        fname = os.path.join(self.outdir, 'w__{}__{}.fits'.format(mask1, mask2))
        w = nmt.NmtWorkspace()
        if not os.path.isfile(fname):
            n_iter = self.data['healpy']['n_iter']
            w.compute_coupling_matrix(self.f1, self.f2, self.b,
                                      n_iter=n_iter)
            w.write_to(fname)
        else:
            w.read_from(fname)
        return w

    def get_ell_cl(self):
        fname = os.path.join(self.outdir, 'cl_{}_{}.npz'.format(self.tr1, self.tr2))
        ell = self.b.get_effective_ells()
        if not os.path.isfile(fname):
            f1, f2 = self.load_fields()
            w = self.get_workspace()
            cl = w.decouple_cell(nmt.compute_coupled_cell(f1, f2))
            np.savez(fname, ell=ell, cl=cl)
        else:
            cl_file = np.load(fname)
            cl = cl_file['cl']
            if np.any(ell != cl_file['ell']):
                raise ValueError('The file {} does not have the expected bpw. Aborting!'.format(fname))

        return ell, cl


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('tr1', type=str, help='Tracer 1 name')
    parser.add_argument('tr2', type=str, help='Tracer 2 name')
    args = parser.parse_args()

    cl = Cl(args.INPUT, args.tr1, args.tr2)
