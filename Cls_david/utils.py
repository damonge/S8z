import numpy as np
import os
import healpy as hp
import pymaster as nmt
import pyccl as ccl


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

    def get_systematic_map(self, name, return_mask=False):
        if self.type == 'gc':
            m = hp.read_map(self.get_path(self.config_h['systematics'][name]['file']),
                            verbose=False)
            self._check_size(m)
            msk = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
            msk_good = msk > self.config['mask_cut_gc']
            msk[~msk_good] = 0
            m_mean = np.sum(msk*m)/np.sum(msk)
            m[msk_good] = m[msk_good] - m_mean
            m[~msk_good] = 0
            mp = [m]
        elif self.type == 'sh':
            msk  = hp.read_map(self.fname_msk, verbose=False) 
            self._check_size(msk)
            we1 = hp.read_map(self.get_path(self.config_h['prefix'] + 'w'
                                            + self.config_h['systematics'][name]['file'] +
                                            'e1_ns%d.fits' % self.nside ),
                              verbose=False)
            self._check_size(we1)
            we2 = hp.read_map(self.get_path(self.config_h['prefix'] + 'w'
                                            + self.config_h['systematics'][name]['file'] +
                                            'e2_ns%d.fits' % self.nside ),
                              verbose=False)
            self._check_size(we2)
            ip_good = msk > 0
            q = np.zeros_like(we1)
            q[ip_good] = -we1[ip_good] / msk[ip_good]
            u = np.zeros_like(we2)
            u[ip_good] = we2[ip_good] / msk[ip_good]
            mp = [q, u]

        if return_mask:
            return mp, msk
        else:
            return mp

    def get_spin(self):
        if self.type == 'gc':
            return 0
        elif self.type == 'sh':
            return 2
        return 0

    def get_path(self, fn):
        return os.path.join(self.config['predir_in'], fn)

    def get_ccl_tracer(self, cosmo):
        _, z, _, nz = np.loadtxt(self.get_path(self.config_h['dndz']),
                                 unpack=True)
        if self.type == 'gc':
            bz = np.ones_like(z) * self.config_h['bias']
            tr = ccl.NumberCountsTracer(cosmo, False, (z, nz), (z, bz))
        elif self.type == 'sh':
            tr = ccl.WeakLensingTracer(cosmo, (z, nz))
        return tr

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

    def get_mask(self):
        if self.type == 'gc':
            msk  = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
            msk_good = msk > self.config['mask_cut_gc']
            msk[~msk_good] = 0
            return msk
        elif self.type == 'sh':
            msk = hp.read_map(self.fname_msk, verbose=False)
            self._check_size(msk)
        return msk

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
            return np.array([np.ones(3*self.nside) * np.mean(msk) / ndens])
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
            return np.array([nl_c, 0*nl_c, 0*nl_c, nl_c])

    def get_systematic_field(self, name):
        mp, msk = self.get_systematic_map(name, return_mask=True)
        return nmt.NmtField(msk, mp, n_iter=self.config['n_iter'])

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
            mp = [delta]
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
            mp = [q, u]

        if 'systematics' in self.config_h:
            temp = []
            for n, d in self.config_h['systematics'].items():
                if d.get('deproject', False):
                    temp.append(self.get_systematic_map(n, return_mask=False))
            if len(temp) == 0:
                temp = None
        else:
            temp = None

        field = nmt.NmtField(msk, mp, templates=temp,
                             n_iter=self.config['n_iter'])
        return field


class Cell(object):
    def __init__(self, tracers, config, bins):
        spin_d = {'gc': 0, 'sh': 2}
        nmaps = [1, 2]
        self.config = config
        self.tracers = tracers
        self.name = tracers[0]+'_'+tracers[1]
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

    def _get_l_arr_interp(self):
        return np.unique(np.geomspace(2, 3*self.nside).astype(int)).astype(float)

    def _get_cl_interpolated(self, cosmo, fields):
        from scipy.interpolate import interp1d

        lt = self._get_l_arr_interp()
        t1 = fields[self.tracers[0]].get_ccl_tracer(cosmo)
        t2 = fields[self.tracers[1]].get_ccl_tracer(cosmo)
        clt = ccl.angular_cl(cosmo, t1, t2, lt)
        cli = interp1d(np.log(lt), clt, fill_value=(0, clt[-1]), bounds_error=False)
        l_out = np.arange(3*self.nside)
        cl_out = np.zeros(3*self.nside)
        cl_out[1:] = cli(np.log(l_out[1:]))
        return cl_out

    def get_cl_signal(self, cosmo, fields):
        cl_th = np.zeros([self.ncls, 3*self.nside])
        cl_th[0, :] = self._get_cl_interpolated(cosmo, fields)
        return cl_th

    def get_cl_theory(self, cosmo, fields, add_noise=True, return_decoupled=True):
        w = self.get_workspace(fields)

        # Compute coupled theory spectrum
        clt = self.get_cl_signal(cosmo, fields)
        # Sometimes the MCM won't go up to 3*nside
        clt[:, :w.wsp.lmax+1] = w.couple_cell(clt)
        clt[:, w.wsp.lmax:] = clt[:, w.wsp.lmax][:, None]

        if add_noise:
            # Compute coupled N_ell if needed
            if self.tracers[0] == self.tracers[1]:
                nl = fields[self.tracers[0]].get_nl_coupled()
                clt += nl

        # Decouple if needed (useful for plotting)
        if return_decoupled:
            clt = w.decouple_cell(clt)
        else:  # Otherwise just divide by the mean mask product (useful for covariances)
            msk1 = fields[self.tracers[0]].get_mask()
            if self.tracers[0] == self.tracers[1]:
                msk2 = msk1
            else:
                msk2 = fields[self.tracers[1]].get_mask()
            maskprod_mean = np.mean(msk1*msk2)
            clt /= maskprod_mean
        return clt

    def load_spectra(self):
        d = np.load(self.get_output_prefix()+'.npz')
        if np.all(d['ell'] != self.ells):
            raise ValueError("Input file is incompatible")
        self.c_ell = d['cl']
        self.n_ell = d['nl']
        self.n_ell_analytic = d['nla']
        self.c_ell_sys = {}
        for k in d.keys():
            if k.startswith('cl_sys'):
                self.c_ell_sys[k[6:]] = d[k]

    def compute_spectra(self, fields, save=True):
        if os.path.isfile(self.get_output_prefix()+'.npz') and (not self.recompute):
            self.load_spectra()
        else:
            self.get_workspace(fields)
            self.c_ell = self.get_cl(fields)
            self.n_ell = self.get_nl(fields)
            self.n_ell_analytic = self.get_nl(fields, use_analytic=True)
            self.c_ell_sys = self.get_xsys(fields)
            if save:
                save_data = {'ell': self.ells,
                             'cl': self.c_ell,
                             'nl':self.n_ell,
                             'nla': self.n_ell_analytic}
                for n, c in self.c_ell_sys.items():
                    save_data['cl_sys_' + n] = c
                np.savez(self.get_output_prefix(),
                         **save_data)

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

    def get_xsys(self, fields):
        if fields[self.tracers[0]].name != fields[self.tracers[1]].name:
            return {}

        xsys = {}
        f = fields[self.tracers[0]]
        if 'systematics' in f.config_h:
            fl1 = f.get_field()
            w = self.get_workspace(fields)
            for n, d in f.config_h['systematics'].items():
                if d['xcorr']:
                    fl2 = f.get_systematic_field(n)
                    cl = w.decouple_cell(nmt.compute_coupled_cell(fl1, fl2))
                    xsys[n]=cl
        return xsys

    def get_nl(self, fields, use_analytic=False):
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


class Covar(object):
    def __init__(self, d1, d2, config):
        self.name = d1[0]+'_'+d1[1]+'_x_'+d2[0]+'_'+d2[1]
        self.config = config
        self.tracers_1 = d1
        self.tracers_2 = d2
        self.msks_1 = [config['maps'][t]['mask']
                       for t in self.tracers_1]
        self.msks_2 = [config['maps'][t]['mask']
                       for t in self.tracers_2]
        self.recompute = config['recompute']

    def get_output_prefix(self):
        return os.path.join(self.config['predir_out'],
                            'cov_'+self.name)

    def _get_workspace_file(self, m1, m2):
        for order1 in [-1, 1]:
            ms1 = m1[::order1]
            st1 = ms1[0]+'_'+ms1[1]
            for order2 in [-1, 1]:
                ms2 = m2[::order1]
                st2 = ms2[0]+'_'+ms2[1]
                fname = os.path.join(self.config['predir_out'],
                                     'cwsp_'+st1+'_'+st2+'.fits')
                if os.path.isfile(fname):
                    return fname, True
        return fname, False
            
    def get_workspace(self, fields):
        fname, found = self._get_workspace_file(self.msks_1,
                                                self.msks_2)
        cw =  nmt.NmtCovarianceWorkspace()
        if found and (not self.recompute):
            cw.read_from(fname)
        else:
            print("Computing " + fname)
            cw.compute_coupling_coefficients(fields[self.tracers_1[0]].get_field(),
                                             fields[self.tracers_1[1]].get_field(),
                                             fields[self.tracers_2[0]].get_field(),
                                             fields[self.tracers_2[1]].get_field())
            cw.write_to(fname)
        return cw

    def _search_cl(self, cls, n1, n2):
        if n1+'_'+n2 in cls:
            return cls[n1+'_'+n2]
        elif n2+'_'+n1 in cls:
            return cls[n2+'_'+n1]
        raise ValueError(f"Combination {n1}-{n2} doesn't exist")

    def compute_covariance(self, cosmo, fields, cls, save=True):
        if os.path.isfile(self.get_output_prefix()+'.npz') and (not self.recompute):
            d = np.load(self.get_output_prefix()+'.npz')
            self.cov = d['cov']
        else:
            self.cov = self.get_cov(cosmo, fields, cls)
            if save:
                np.savez(self.get_output_prefix(), cov=self.cov)

    def get_cov(self, cosmo, fields, cls):
        cw = self.get_workspace(fields)
        cla1b1 = self._search_cl(cls,
                                 self.tracers_1[0],
                                 self.tracers_2[0]).get_cl_theory(cosmo, fields,
                                                                  return_decoupled=False)
        cla1b2 = self._search_cl(cls,
                                 self.tracers_1[0],
                                 self.tracers_2[1]).get_cl_theory(cosmo, fields,
                                                                  return_decoupled=False)
        cla2b1 = self._search_cl(cls,
                                 self.tracers_1[1],
                                 self.tracers_2[0]).get_cl_theory(cosmo, fields,
                                                                  return_decoupled=False)
        cla2b2 = self._search_cl(cls,
                                 self.tracers_1[1],
                                 self.tracers_2[1]).get_cl_theory(cosmo, fields,
                                                                  return_decoupled=False)
        cla = self._search_cl(cls, self.tracers_1[0], self.tracers_1[1])
        wa = cla.get_workspace(fields)
        ncla = cla.ncls
        nl = cla.nl
        clb = self._search_cl(cls, self.tracers_2[0], self.tracers_2[1])
        wb = clb.get_workspace(fields)
        nclb = clb.ncls

        cov = nmt.gaussian_covariance(cw,
                                      fields[self.tracers_1[0]].get_spin(),
                                      fields[self.tracers_1[1]].get_spin(),
                                      fields[self.tracers_2[0]].get_spin(),
                                      fields[self.tracers_2[1]].get_spin(),
                                      cla1b1, cla1b2, cla2b1, cla2b2, wa, wb)
        cov = cov.reshape([nl, ncla, nl, nclb])
        return cov
