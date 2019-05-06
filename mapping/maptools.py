import healpy as hp
import numpy as np

def get_fits_iterator(fname, colnames, hdu=1, nrows_per_chunk=None):
    import fitsio

    # Open file and select HDU
    fitf = fitsio.FITS(fname, mode='r')
    tab = fitf[hdu]

    # Count rows and number of chunks
    nrows = tab.get_nrows()
    # If None, then just one chunk
    if nrows_per_chunk is None:
        nrows_per_chunk = nrows
    # Add one chunk if there are leftovers
    nchunks = nrows // nrows_per_chunk
    if nrows_per_chunk * nchunks < nrows:
        nchunks += 1

    for i in range(nchunks):
        start = i * nrows_per_chunk
        end = min((i + 1) * nrows_per_chunk, nrows)
        data = tab.read_columns(colnames,
                                rows=range(start, end))
        yield data

def get_weighted_maps(iterator, nside, name_ra, name_dec,
                      name_weight=None, name_field=None,
                      masks=None):

    npix = hp.nside2npix(nside)

    if masks is None:
        nmaps = 1
        masks = [['all']]
    else:
        nmaps = len(masks)

    map_counts = np.zeros([nmaps, npix])
    if name_weight is not None:
        map_weights = np.zeros([nmaps, npix])
    else:
        map_weights = None
    if name_field is not None:
        map_field = np.zeros([nmaps, npix])
    else:
        map_field = None

    for d in iterator:
        ipix = hp.ang2pix(nside,
                          np.radians(90 - d[name_dec]),
                          np.radians(d[name_ra]))
        print(len(ipix))
        for im, m in enumerate(masks):
            mask = np.ones(len(ipix), dtype=bool)
            if m[0] == 'tag':
                mask = mask & (d[m[1]] == m[2])
            elif m[0] == 'range':
                mask = mask & (d[m[1]] < m[3]) & (d[m[1]] >= m[2])

            ip = ipix[mask]
            map_counts[im, :] += np.bincount(ip,
                                             minlength=npix)
            if name_weight is not None:
                w = d[name_weight][mask]
                map_weights[im, :] += np.bincount(ip,
                                                  minlength=npix,
                                                  weights=w)
                if name_field is not None:
                    f = d[name_field][mask]
                    map_field[im, :] += np.bincount(ip,
                                                    minlength=npix,
                                                    weights=w * f)
            else:
                if name_field is not None:
                    f = d[name_field][mask]
                    map_field[im, :] += np.bincount(ip,
                                                    minlength=npix,
                                                    weights=f)

    if nmaps == 1:
        map_counts = map_counts.flatten()
        if map_weights is not None:
            map_weights = map_weights.flatten()
        if map_field is not None:
            map_field = map_field.flatten()

    return map_counts, map_weights, map_field
