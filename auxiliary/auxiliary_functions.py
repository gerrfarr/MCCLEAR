import healpy as hp
import numpy as np
from scipy import stats
import yaml

def read_config_default_vals(config_path, args, sys_argv=[]):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract argument names from sys_argv
    provided_args = {arg.lstrip('-').split('=')[0] for arg in sys_argv}

    for key in config.keys():
        if key in vars(args) and key not in provided_args:
            setattr(args, key, config[key])

    return args

def parse_ranges(range_str):
    ranges = range_str.split(',')
    indices = []
    for r in ranges:
        start, end = map(int, r.split('-'))
        indices.extend(range(start, end + 1))
    return indices


def read_pixell_map_as_hp(path, nside):
    from pixell import enmap
    map = enmap.read_map(path)
    return map.to_healpix(nside = nside)


def phi2kappa(phi_alm):
    lmax = hp.Alm.getlmax(phi_alm.size)
    l = np.arange(lmax + 1)
    return hp.almxfl(alm=phi_alm, fl=l * (l + 1) / 2)


def read_mask(path, nside):
    try:
        mask = hp.ud_grade(hp.read_map(path), nside)
    except ValueError as ex:
        mask = read_pixell_map_as_hp(path, nside)
    return mask


def trim_or_pad_cls(arr, target_length, pad_value=np.nan):
    current_length = len(arr)

    if current_length > target_length:
        # Trim the array
        return arr[:target_length]
    elif current_length < target_length:
        # Pad the array
        padded_array = np.full(target_length, pad_value)
        padded_array[:current_length] = arr
        return padded_array
    else:
        # If the length is already equal to the target length, return the array as is
        return arr


def bin_spectrum(ells, cells, bin_edges, ell_weighted=False):
    binned_cells = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_edges) - 1):
        bin_mask = (ells >= bin_edges[i]) & (ells < bin_edges[i + 1])
        if np.any(bin_mask):
            binned_cells[i] = np.mean(cells[bin_mask]) if not ell_weighted else np.sum(ells[bin_mask] * cells[bin_mask]) / np.sum(ells[bin_mask])
        else:
            binned_cells[i] = np.nan  # or some other placeholder for empty bins

    return binned_cells