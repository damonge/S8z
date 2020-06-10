import numpy as np
import os
import healpy as hp
import pymaster as nmt


class Field(object):
    def __init__(self, config, name):
        self.config = config
        self.config_h = self.config['maps'][name]
        self.name = name
        self.type = self.config_h['type']
        self.fname_msk = self.get_path(self.config['masks'][self.config_h['mask']])
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)

    def _check_size(self, m):
        if len(m) != self.npix:
            raise ValueError("Inconsistent map %d %d", len(m), self.npix)

    def get_spin(self):
        if self.type == 'gc':
            return 0
        elif self.type == 'sh':
            return 2
        return 0

    def get_path(self, fn):
        return os.path.join(self.config['predir_in'], fn)

    def get_field(self, i_field=None):
        if i_field is None:
            if not hasattr(self, 'field'):
                self.field = self.make_field()
            return self.field
        else:
            if self.type == 'gc':
                return self.get_field(None)
            elif self.type == 'sh':
                return self.make_field(i_field)

    def get_nl_coupled(self):
        if self.type == 'gc':
            msk  = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
            nc = hp.read_map(self.get_path(self.config_h['wcounts']), verbose=False)
            self._check_size(nc)
            msk_good = msk > self.config['mask_cut_gc']
            msk[~msk_good] = 0
            nmean = np.sum(nc[msk_good]) / np.sum(msk[msk_good])
            ndens = nmean * self.npix / (4*np.pi)
            return [np.ones(3*self.nside) * np.mean(msk) / ndens]
        elif self.type == 'sh':
            msk = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
            ip_bad = msk <= 0

            sums = np.load(self.get_path(self.config_h['sums']))
            w2s2mean = sums['w2s2'] / self.npix
            opm_mean = sums['wopm'] / sums['w']

            # N_l
            pix_area = 4*np.pi/self.npix
            nl_c = np.ones(3*self.nside)*pix_area*w2s2mean/opm_mean**2
            return [nl_c, 0*nl_c, 0*nl_c, nl_c]

    def make_field(self, i_field=None):
        if self.type == 'gc':
            msk = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
            nc = hp.read_map(self.get_path(self.config_h['wcounts']), verbose=False)
            self._check_size(nc)
            msk_good = msk > self.config['mask_cut_gc']            
            msk[~msk_good] = 0
            nmean = np.sum(nc[msk_good]) / np.sum(msk[msk_good])
            delta = np.zeros_like(msk)
            delta[msk_good] = nc[msk_good] / (nmean * msk[msk_good]) - 1
            field = nmt.NmtField(msk, [delta], n_iter=self.config['n_iter'])
        elif self.type == 'sh':
            if i_field is not None:
                mid = 'rot%d_' % i_field
            else:
                mid = ''
            msk  = hp.read_map(self.fname_msk, verbose=False) 
            self._check_size(msk)
            we1 = hp.read_map(self.get_path(self.config_h['prefix'] + mid +
                                            'we1_ns%d.fits' % self.nside ),
                              verbose=False)
            self._check_size(we1)
            we2 = hp.read_map(self.get_path(self.config_h['prefix'] + mid +
                                            'we2_ns%d.fits' % self.nside ),
                              verbose=False)
            self._check_size(we2)
            sums = np.load(self.get_path(self.config_h['sums']))
            ip_good = msk > 0
            opm_mean = sums['wopm'] / sums['w']
            q = np.zeros_like(we1)
            q[ip_good] = -we1[ip_good] / (msk[ip_good] * opm_mean)
            u = np.zeros_like(we2)
            u[ip_good] = we2[ip_good] / (msk[ip_good] * opm_mean)
            field = nmt.NmtField(msk, [q, u])
        return field


class Cell(object):
    def __init__(self, d, config, bins):
        spin_d = {'gc': 0, 'sh': 2}
        nmaps = [1, 2]
        self.config = config
        self.tracers = d['tracers']
        self.name = d['name']
        self.msks = [config['maps'][t]['mask']
                     for t in self.tracers]
        self.bins = bins
        self.ells = self.bins.get_effective_ells()
        self.nl = len(self.ells)
        self.spins = [spin_d[config['maps'][t]['type']]
                      for t in self.tracers]
        self.ncls = np.prod([max(s, 1) for s in self.spins])
        self.shape = [self.ncls, self.nl]
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.recompute = config['recompute']

    def load_spectra(self):
        d = np.load(self.get_output_prefix()+'.npz')
        if np.all(d['ell'] != self.ells):
            raise ValueError("Input file is incompatible")
        self.c_ell = d['cl']
        self.n_ell = d['nl']
        self.n_ell_analytic = d['nla']

    def compute_spectra(self, fields, save=True):
        if os.path.isfile(self.get_output_prefix()+'.npz') and (not self.recompute):
            self.load_spectra()
        else:
            self.get_workspace(fields)
            self.c_ell = self.get_cl(fields)
            self.n_ell = self.get_nl(fields, use_analytic=True)
            self.n_ell_analytic = self.get_nl(fields, use_analytic=True)
            if save:
                np.savez(self.get_output_prefix(),
                         ell=self.ells,
                         cl=self.c_ell,
                         nl=self.n_ell,
                         nla=self.n_ell_analytic)

    def get_output_prefix(self):
        return os.path.join(self.config['predir_out'],
                            'cl_'+self.name)

    def _get_workspace_file(self, m):
        for order in [-1, 1]:
            ms = m[::order]
            fname = os.path.join(self.config['predir_out'],
                                 'wsp_'+ms[0]+'_'+ms[1]+'.fits')
            if os.path.isfile(fname):
                return fname, True
        return fname, False

    def get_workspace(self, fields):
        fname, found = self._get_workspace_file(self.msks)
        w = nmt.NmtWorkspace()
        if found and (not self.recompute):
            w.read_from(fname)
        else:
            print("Computing " + fname)
            w.compute_coupling_matrix(fields[self.tracers[0]].get_field(),
                                      fields[self.tracers[1]].get_field(),
                                      self.bins)
            w.write_to(fname)
        return w

    def get_cl(self, fields):
        w = self.get_workspace(fields)
        cl = w.decouple_cell(nmt.compute_coupled_cell(fields[self.tracers[0]].get_field(),
                                                      fields[self.tracers[1]].get_field()))
        return cl

    def get_nl(self, fields, i_field=None, use_analytic=False):
        if fields[self.tracers[0]].name != fields[self.tracers[1]].name:
            return np.zeros(self.shape)

        f = fields[self.tracers[0]]
        w = self.get_workspace(fields)
        if f.type == 'gc':
            nl = w.decouple_cell(f.get_nl_coupled())
        elif f.type == 'sh':
            if use_analytic:
                nl = w.decouple_cell(f.get_nl_coupled())
            else:
                cls = []
                for irot in range(f.config_h['nrot']):
                    print(" - %d-th/%d rotation" % (irot, f.config_h['nrot']))
                    try:
                        f_nmt = f.get_field(irot)
                        cl = w.decouple_cell(nmt.compute_coupled_cell(f_nmt, f_nmt))
                        cls.append(cl)
                    except:
                        print("File not found. Oh well...")
                cls = np.array(cls)
                nl = np.mean(cls, axis=0)

        return nl
