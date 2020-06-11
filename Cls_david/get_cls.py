import numpy as np
import pymaster as nmt
import yaml
import sys
import utils as ut
import matplotlib.pyplot as plt
import os
import pyccl as ccl


fname_pars = sys.argv[1]

with open(fname_pars) as f:
    p = yaml.safe_load(f)

os.system('mkdir -p ' + p['predir_out'])

l_edges = np.array(p['bpw_edges'])
l_edges = l_edges[l_edges <= 3*p['nside']]
print(l_edges)
if 3*p['nside'] not in l_edges:
    l_edges = np.append(l_edges, 3*p['nside'])

b = nmt.NmtBin.from_edges(l_edges[:-1], l_edges[1:])
l_eff = b.get_effective_ells()

fields = {}
for name in p['maps']:
    fields[name] = ut.Field(p, name)
nfields = len(fields.keys())
i_field = 0
i_shear = 0
field_ids = {}
field_shear_ids = {}
for n in fields.keys():
    field_ids[n] = i_field
    if fields[n].type == 'sh':
        field_shear_ids[n] = i_shear
        i_shear += 1
    i_field += 1
nfields = i_field
nfields_shear = i_shear
field_names = sorted(p['maps'].keys())

cls = {}
for i1, m1 in enumerate(field_names):
    for m2 in field_names[i1:]:
        cell = ut.Cell([m1, m2], p, b)
        print(cell.name)
        cell.compute_spectra(fields)
        cls[cell.name] = cell
exit(1)
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81)
covs = {}
for i1, cl1 in enumerate(p['c_ells']):
    for cl2 in p['c_ells'][i1:]:
        cov = ut.Covar(cl1, cl2, p)
        print(cov.name)
        cov.compute_covariance(cosmo, fields, cls)
        covs[cov.name] = cov
fig, axes = plt.subplots(nfields, nfields, sharex=True, figsize=(10, 10))
for cl in p['c_ells']:
    cell = cls[cl[0]+'_'+cl[1]]
    cov = covs[cl[0]+'_'+cl[1]+'_x_'+cl[0]+'_'+cl[1]]
    i1 = field_ids[cell.tracers[0]]
    i2 = field_ids[cell.tracers[1]]
    ax = axes[i1][i2]
    ax.errorbar(cell.ells, cell.c_ell[0],
                yerr=np.sqrt(np.diag(cov.cov[:,0,:,0])),
                fmt='r-')
    #ax.plot(cell.ells, -cell.c_ell[0], 'r--')
    ax.plot(cell.ells, cell.get_cl_theory(cosmo, fields)[0], 'b-')
    if not np.all(cell.n_ell == 0):
        ax.plot(cell.ells, cell.n_ell[0], 'k-')
    ax.set_title(cell.name)
    ax.set_xscale('log')
    ax.set_yscale('log')
'''
for ipol in [0, 1, 2, 3]:
    fig, axes = plt.subplots(nfields_shear, nfields_shear, sharex=True, figsize=(10, 10))
    for cl in p['c_ells']:
        cell = cls[cl['name']]
        if ((fields[cell.tracers[0]].type!='sh') or
            (fields[cell.tracers[1]].type!='sh')):
            continue
        i1 = field_shear_ids[cell.tracers[0]]
        i2 = field_shear_ids[cell.tracers[1]]
        ax = axes[i1][i2]
        #ax.plot(cell.ells, cell.c_ell[ipol] - cell.n_ell[ipol], 'k-')
        #ax.plot(cell.ells, np.zeros_like(cell.ells), 'k--')
        ax.plot(cell.ells, cell.c_ell[ipol], 'k-')
        ax.plot(cell.ells, cell.n_ell[ipol], 'r-')
        ax.plot(cell.ells, cell.n_ell_analytic[ipol], 'b-')
        ax.set_title(cell.name)
        ax.set_xscale('log')
'''
plt.show()
