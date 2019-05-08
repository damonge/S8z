# S8(z)

## Data
The derived data products (maps, masks, redshift distributions and possibly other metadata), can be found in `/mnt/bluewhale/damonge/S8z_data/derived_products`
- For DES galaxy clustering, look at the subdirectory `des_clustering`. There'll be redshift distributions (e.g. `dndz_binX.txt`) and maps with number counts (e.g. `map_counts_w_binX_ns4096.fits`), as well as the corresponding mask (`mask_ns4096.fits`).
- For DES galaxy shear, look at the subdirectory `des_shear`. All files are duplicated for the two different shape catalogs (`metacal` and `im3shape`). Redshift distributions in `dndz_metacal_binX.txt`. Maps with weighted number counts in `map_metacal_binX_counts_w_ns4096.fits`, weighted ellipticities in `map_metacal_binX_counts_{e1,e2}_ns4096.fits` and weighted multiplicative bias in `map_metacal_binX_counts_opm_ns4096.fits`.
- Best-fit DES parameters (from their papers) can be found in `des_clustering/bf_params.txt`.
