import os

file_path = os.path.abspath(os.path.dirname(__file__))
import argparse
from auxiliary import read_config_default_vals

parser = argparse.ArgumentParser()
# what to do
parser.add_argument('--sim_ids', action="store", dest='sim_ids', default="0-399", type=str, help='Range of simulation IDs to use, can be a range separated by a dash or several ranges separated by a comma.')
parser.add_argument('--n_sim_offset', action="store", dest='n_sim_offset', default=None, type=int, help='Offset between input and reconstruction simulation IDs (e.g., if input simulations are 0-399 and reconstruction simulations are 1000-1399, n_sim_offset=1000).')
parser.add_argument("--planck_sims", dest='planck_sims', action='store_true', default=False, help='Use Planck lensing reconstruction simulations.')

parser.add_argument('--use_mpi', action="store_true", dest='use_mpi', default=False, help='Use MPI for parallel processing.')
parser.add_argument('--config_path', action="store", dest='config_path', default=file_path+"config.yml", type=str, help='Path to the configuration file.')

parser.add_argument('--output_dir', action="store", dest='output_dir', default="", type=str, help='Directory to save the output files.')
parser.add_argument('--output_prefix', action="store", dest='output_prefix', default="", type=str, help='Prefix for the output files.')

parser.add_argument('--kappa_recon_sims_path', action="store", dest='kappa_sims_path', type=str, default=None, help='Path to the kappa reconstruction simulations.')
parser.add_argument('--kappa_recon_sims_prefix', action="store", dest='kappa_sims_prefix', type=str, default="", help='Prefix for the kappa reconstruction simulation files.')
parser.add_argument('--kappa_recon_sims_suffix', action="store", dest='kappa_sims_suffix', type=str, default="", help='Suffix for the kappa reconstruction simulation files.')
parser.add_argument('--kappa_recon_id_format', action="store_true", dest='kappa_recon_id_format', default=None, help='Format for the kappa reconstruction simulation IDs.')
parser.add_argument('--kappa_input_sims_path', action='store', dest='signal_only_kappa_path', default=None, help='Path to the input kappa simulations.')
parser.add_argument('--kappa_input_sims_prefix', action="store", dest='kappa_sims_prefix', type=str, default="", help='Prefix for the input kappa simulation files.')
parser.add_argument('--kappa_input_sims_suffix', action="store", dest='kappa_sims_suffix', type=str, default="", help='Suffix for the input kappa simulation files.')
parser.add_argument('--kappa_input_id_format', action="store_true", dest='kappa_recon_id_format', default='', help='Format for the input kappa simulation IDs, provided in as python format string (e.g., \'{:04}\' for four digits with leading zeros).')

parser.add_argument('--kappa_mask_path', action="store", dest='kappa_mask_path', type=str, default=None, help='Path to the kappa mask file.')
parser.add_argument('--lss_mask_path', action="store", dest='lss_mask_path', type=str, default=None, help='Path to the LSS mask file.')
parser.add_argument('--extra_mask', action="store", dest='extra_mask', type=str, default=None, help='Path to an additional mask file.')

parser.add_argument('--kappa_mean_field_path', action='store', dest='kappa_mean_field_path', default=None, help='Path to the kappa mean field file.')

parser.add_argument('--nside', action="store", dest='nside', default=None, type=int, help='Resolution parameter for HEALPix maps.')
parser.add_argument('--lmax', action="store", dest='lmax', default=4000, type=int, help='Maximum multipole for spherical harmonics.')
parser.add_argument('--rotate_kappa_alm', action='store_true', dest='rotate_input_kappa', default=None, help='Rotate kappa alm from galactic to equatorial coordinates.')
parser.add_argument('--mask_kappa_recon', action="store_true", dest='mask_kappa_recon', default=False, help='Apply mask to kappa reconstruction.')
parser.add_argument('--kappa_mask_is_cmb_mask', action="store_true", dest='kappa_mask_is_cmb_mask', default=False, help='Indicate that the provided kappa mask is the CMB mask, and the effective kappa mask is the square of this mask.')
parser.add_argument('--use_joined_mask_kappa', action="store_true", dest='use_joined_mask', default=False, help='Use the joined mask for kappa.')
parser.add_argument('--use_joined_mask_lss', action="store_true", dest='use_joined_mask', default=False, help='Use the joined mask for LSS.')
parser.add_argument('--use_namaster', action="store_true", dest='use_namaster', default=False, help='Use NaMaster for power spectrum estimation.')
args = parser.parse_args()

args = read_config_default_vals(args.config_path, args)

if args.use_mpi:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Detected MPI run with size {size}. This is rank {rank}.")
    print(f"This process has {len(os.sched_getaffinity(0))} CPUs available.")
else:
    rank = 0
    size = 1

import healpy as hp
import numpy as np
from auxiliary import parse_ranges, phi2kappa, read_mask

if args.use_namaster:
    import pymaster as nmt

## load and process masks
mask = read_mask(args.kappa_mask_path, args.nside)

if not args.apply_kappa_mask:
    kappa_mask = mask**2  # squared because this is the CMB map and kappa is quadratic in the mask
else:
    kappa_mask = mask

if args.lss_mask_path is not None:
    lss_mask = read_mask(args.lss_mask_path, args.nside)
else:
    lss_mask = kappa_mask

if args.extra_mask is not None:
    extra_mask = read_mask(args.extra_mask, args.nside)
else:
    extra_mask = np.ones_like(kappa_mask)

joined_mask = kappa_mask * lss_mask

kappa_mask *= extra_mask
lss_mask *= extra_mask
joined_mask *= extra_mask

kappa_mask_final = joined_mask if args.use_joined_mask_kappa else kappa_mask
lss_mask_final = joined_mask if args.use_joined_mask_lss else lss_mask

w_fac_kg = np.mean(kappa_mask_final * lss_mask_final)

## load mean field
if args.kappa_mean_field_path is not None:
    mean_field = hp.read_alm(args.kappa_mean_field_path)
else:
    mean_field = 0


## set up sim ids to process

sim_ids2process = None
sim_ids_split = None
if rank == 0:
    all_sim_ids = parse_ranges(args.sim_ids)
    if args.use_mpi:
        sim_ids_split = np.array_split(all_sim_ids, size)
    else:
        sim_ids_split = all_sim_ids

if not args.use_mpi:
    sim_ids2process = comm.scatter(sim_ids_split, root=0)

cl_outputs = np.ones((len(sim_ids2process), 2,  args.lmax + 1))

if args.rotate_kappa_alm:
    hp_rot_gc = hp.rotator.Rotator(coord=["G", "C"])

for n, i in enumerate(sim_ids2process):
    print(f"Reading kappa sim {i}")

    recon_alm = hp.read_alm(args.kappa_sims_path+args.kappa_sims_prefix + f"{i+args.n_sim_offset:{args.kappa_recon_id_format}}{args.kappa_sims_suffix}.fits").astype('complex')  # reconstructed kappa
    if args.rotate_kappa_alm:
        print(f"Rotating sim {i} reconstruction into equatorial coordinates...")
        recon_alm = hp_rot_gc.rotate_alm(recon_alm)

    lmax = hp.Alm.getlmax(recon_alm.size)

    recon_alm -= mean_field
    recon_alm[~np.isfinite(recon_alm)] = 0
    if args.mask_kappa_recon:
        recon_alm = hp.map2alm(hp.alm2map(recon_alm, args.nside) * kappa_mask_final, lmax=lmax)
    elif not args.mask_kappa_recon and args.use_joined_mask_kappa:
        recon_alm = hp.map2alm(hp.alm2map(recon_alm, args.nside) * lss_mask, lmax=lmax)
    elif args.extra_mask is not None:
        recon_alm = hp.map2alm(hp.alm2map(recon_alm, args.nside) * extra_mask, lmax=lmax)


    print(f"Loading signal for sim {i}...")
    input_kappa_alm_raw = phi2kappa(hp.read_alm(args.signal_only_kappa_path + f"{i:{args.kappa_input_id_format}}.fits", hdu=4).astype('complex'))
    if args.rotate_kappa_alm:
        print(f"Rotating sim {i} input into equatorial coordinates...")
        input_kappa_alm_raw = hp_rot_gc.rotate_alm(input_kappa_alm_raw)

    input_kappa_alm = hp.alm2map(input_kappa_alm_raw, args.nside)
    input_kappa_alm_kmask = hp.map2alm(input_kappa_alm * kappa_mask_final, lmax=lmax)
    input_kappa_alm_lssmask = hp.map2alm(input_kappa_alm * lss_mask_final, lmax=lmax)

    cl_outputs[n, 0, :lmax+1] = hp.alm2cl(recon_alm, input_kappa_alm_lssmask)[:args.lmax+1] / w_fac_kg
    cl_outputs[n, 1, :lmax+1] = hp.alm2cl(input_kappa_alm_kmask, input_kappa_alm_lssmask)[:args.lmax+1] / w_fac_kg


if args.use_mpi:
    cl_outputs = comm.gather(cl_outputs, root=0)
    if rank==0:
        cl_outputs = np.concatenate(cl_outputs, axis=0)
        np.savetxt(args.output_dir + args.output_prefix + f"norm_correction_{args.sim_ids}.dat", np.vstack([np.arange(0, args.lmax+1), np.mean(cl_outputs[:,0], axis=0)/np.mean(cl_outputs[:,1], axis=0)]).T, header="ell, norm_correction")