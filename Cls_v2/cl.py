#!/usr/bin/python
import common as co
import numpy as np
import healpy as hp
import pymaster as nmt
import os
import yaml

class Field():
    def __init__(self, data, tr, maps=None, mask=None):
        self.data = co.read_data(data)
        self.tr = tr
        self.spin = self.data['tracers'][tr]['spin']
        self._raw_maps = None
        self._maps = maps
        self._mask = mask
        self.f = self.compute_field()

    def get_field(self):
        return self.f

    def compute_field(self):
        data = self.data
        mask = self.get_mask()
        maps = self.get_maps()
        f = nmt.NmtField(mask, maps, n_iter=data['healpy']['n_iter'])
        return f

    def get_mask(self):
        if self._mask is None:
            data = self.data
            tracer = data['tracers'][self.tr]
            mask = hp.read_map(tracer['mask'])
            mask_good = mask >= tracer['threshold']
            mask[~mask_good] = 0.
            self._mask = mask
        return self._mask

    def get_maps(self):
        if self._maps is None:
            data = self.data
            tracer = data['tracers'][self.tr]
            mask = self.get_mask()
            mask_good = mask >= tracer['threshold']
            # TODO: This can be optimize for the case mask1 == mask2
            raw_maps = self.get_raw_maps()
            if self.spin == 0:
                maps = np.zeros((1, mask.size))
                raw_map = raw_maps[0]
                mean_raw_map = raw_map[mask_good].sum() / mask[mask_good].sum()
                map_dg = np.zeros_like(raw_map)
                map_dg[mask_good] = raw_map[mask_good] / (mean_raw_map * mask[mask_good]) - 1
                maps[0] = map_dg
            else:
                maps = np.zeros((2, mask.size))
                sums = np.load(tracer['sums'])
                map_we1, map_we2 = raw_maps
                opm_mean = sums['wopm'] / sums['w']

                maps[0, mask_good] = (map_we1[mask_good]/mask[mask_good]) / opm_mean
                maps[1, mask_good] = (map_we2[mask_good]/mask[mask_good]) / opm_mean
            self._maps = maps

        return self._maps

    def get_raw_maps(self):
        tracer = self.data['tracers'][self.tr]
        if self._raw_maps is None:
            if self.spin == 0:
                raw_map = hp.read_map(tracer['path'])
                self._raw_maps = np.array([raw_map])
            else:
                map_we1 = hp.read_map(tracer['path1'])
                map_we2 = hp.read_map(tracer['path2'])
                self._raw_maps = np.array([map_we1, map_we2])

        return self._raw_maps



class Cl():
    def __init__(self, data, tr1, tr2):
        self._datapath = data
        self.data = co.read_data(data)
        self.tr1 = tr1
        self.tr2 = tr2
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.b = self.get_NmtBin()
        # Not needed to load cl if already computed
        self._f1 = None
        self._f2 = None
        self._w = None
        ##################
        self.ell, self.cl = self.get_ell_cl()


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

    def get_fields(self):
        if self._f1 is None:
            self._f1 = Field(self._datapath, self.tr1)
            self._f2 = Field(self._datapath, self.tr2)
        return self._f1, self._f2

    def get_workspace(self):
        if self._w is None:
            self._w = self._compute_workspace()
        return self._w

    def _compute_workspace(self):
        mask1 = os.path.basename(self.data['tracers'][self.tr1]['mask'])
        mask2 = os.path.basename(self.data['tracers'][self.tr2]['mask'])
        # Remove the extension
        mask1 = os.path.splitext(mask1)[0]
        mask2 = os.path.splitext(mask2)[0]
        fname = os.path.join(self.outdir, 'w__{}__{}.fits'.format(mask1, mask2))
        w = nmt.NmtWorkspace()
        if not os.path.isfile(fname):
            n_iter = self.data['healpy']['n_iter']
            f1, f2 = self.get_fields()
            w.compute_coupling_matrix(f1.f, f2.f, self.b,
                                      n_iter=n_iter)
            w.write_to(fname)
        else:
            w.read_from(fname)
        return w

    def _compute_noise_gc(self):
        map_ng = self._f1.get_raw_maps()[0]
        map_dg = self._f1.get_maps()[0]
        mask = self._f1.get_mask()
        mask_good = mask > 0  # Already set to 0 all bad pixels in Field()
        npix = mask.size
        nside = hp.npix2nside(npix)

        N_mean = map_ng[mask_good].sum() / mask[mask_good].sum()
        N_mean_srad = N_mean / (4 * np.pi) * npix
        N_ell = mask.sum() / npix / N_mean_srad
        w = self.get_workspace()
        return w.decouple_cell([N_ell * np.ones(3 * nside)])[0]

    def _compute_noise_wl(self):
        pass


    def compute_noise(self):
        tracers = self.data['tracers']
        ell = self.b.get_effective_ells()
        if self.tr1!= self.tr2:
            return 0 * ell
        elif tracers[self.tr1]['spin'] == 0:
            return self._compute_noise_gc()
        else:
            return self._compute_noise_wl()

    def get_ell_cl(self):
        fname = os.path.join(self.outdir, 'cl_{}_{}.npz'.format(self.tr1, self.tr2))
        ell = self.b.get_effective_ells()
        if not os.path.isfile(fname):
            f1, f2 = self.get_fields()
            w = self.get_workspace()
            cl = w.decouple_cell(nmt.compute_coupled_cell(f1.f, f2.f))
            nl = self.compute_noise()
            np.savez(fname, ell=ell, cl=cl-nl, nl=nl)
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
