import healpy as hp
import numpy as np
import yaml

def read_config_default_vals(config_path, args):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for key in config.keys():
        if key in vars(args) and getattr(args, key) is None:
            setattr(args, key, config[key])
        elif key in vars(args) and isinstance(getattr(args, key), bool) and not getattr(args, key) and config[key]:
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
    return hp.almxfl(alm=phi_alm, lfilter=l * (l + 1) / 2)

def read_mask(path, nside):
    try:
        mask = hp.ud_grade(hp.read_map(path), nside)
    except ValueError as ex:
        mask = read_pixell_map_as_hp(path, nside)
    return mask