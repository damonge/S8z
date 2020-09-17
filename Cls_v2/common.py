#!/usr/bin/python
import yaml

def read_data(data):
    with open(data) as f:
        data = yaml.safe_load(f)
    return data

def get_tracers_used(data):
    tracers = []
    for trk, trv in data['cls'].items():
        tr1, tr2 = trk.split('-')
        if trv != 'None':
            tracers.append(tr1)
            tracers.append(tr2)

    tracers_for_cl = []
    for tr in data['tracers'].keys():
        tr_nn = ''.join(s for s in tr if not s.isdigit())
        if tr_nn in tracers:
            tracers_for_cl.append(tr)

    return tracers_for_cl

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


