# MCCLEAR
## (**M**)onte (**C**)arlo (**C**)MB (**L**)ensing (**E**)stim(**A**)tor (**R**)enormalisation

This repository provides tools to compute the normalisation correction for CMB lensing reconstructions in the presence of masks. It can be used to compute the normalisation correction for cross-correlation analyses which use the Planck PR3, Planck PR4 or ACT DR6 lensing reconstructions. 

The code is written in Python and is based on the `healpy` and `numpy` libraries. It includes an option to use the `NaMaster` library to compute the normalisation correction with mode decoupling.

## Requirements

- Python 3.x
- `healpy`
- `numpy`
- `argparse`
- `mpi4py` (optional, for MPI support)
- `NaMaster` (optional, for mode decoupling)

## Usage

To run the normalisation correction computation, use the following command:

```python compute_norm_correction.py --config_path <path_to_config_file>```

Default configuration files are provided in the `configs` directory for Planck and ACT lensing reconstructions. The configuration file specifies the input lensing reconstruction maps, the mask, the output directory, and the normalisation correction parameters. You may override any of the parameters in the configuration file by specifying them as command line arguments.