# --- IMPORTS --- #
import numpy as np

import generate_spectrum as genspec
import helper_functions as helpers
from ic_checks import check_delta_rho, check_elsasser, check_alfven_speed

def box_avg(x, reshape=0):
    # box indicies are always the last 3
    len_shape = len(x.shape)
    if len_shape < 3:
        # either already averaged or not a box quantity
        return x
    axes = tuple(np.arange(len_shape-3, len_shape))
    avg = np.mean(x, axis=axes)
    if reshape:
        shape = avg.shape
        # add back in grid columns for broadcasting
        avg = avg.reshape(*shape, 1, 1, 1)
    return avg


def create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock, athinput,
                             time_lim=1, dt=0.2, expand=False, exp_rate=0., beta=1.0,
                             perp_energy=0.5, amplitude=None,
                             spectrum='isotropic', expo=-5/3, expo_prl=-2.,
                             kpeak=(2,2), kwidth=12.0,
                             amp_rho=0.0,
                             do_truncation=False, n_cutoff=None,
                             do_mode_test=False):

    """
    Generates and saves a simulation setup for Alfven waves in a specified folder.

    Args:
        folder (str): Path to where the output files will be saved.
        h5name (str): The name of the HDF5 file to be created (e.g "output.h5").
        n_X (list): The number of grid points in each dimension (e.g. [nx, ny, nz] = [64, 64, 64]).
        X_min (float): The minimum value of the spatial domain (e.g. [xmin, ymin, zmin] = [0, 0, 0]).
        X_max (float): The maximum value of the spatial domain (e.g. [xmax, ymax, zmax] = [1, 1, 1]).
        meshblock (str): The meshblock configuration for the simulation (e.g. [nx, ny, nz] = [64, 64, 64]).
        athinput: Path to a template athinput file to be overwritten
        time_lim (float, optional): The time limit for the simulation. Defaults to 1.
        dt (float, optional): The time step for the simulation. Defaults to 0.2.
        expand (bool, optional): Whether to turn on expansion. Defaults to False.
        exp_rate (float, optional): Expansion rate for the simulation. Defaults to 0.0.
        beta (float, optional): Defaults to 1.0.
        perp_energy (float, optional): Perpendicular energy scaling factor. Defaults to 0.5.
        spectrum (str, optional): The type of spectrum to generate. Options are ['gaussian', 'isotropic', 'anisotropic']; defaults to 'isotropic'.
        expo (float, optional): Perpendicular exponent for the spectrum. Defaults to -5/3.
        expo_prl (float, optional): Exponent for the parallel spectrum. Defaults to -2.0.
        kpeak (tuple, optional): Peak wavenumber for the gaussian spectrum. Defaults to (kprl, kprp) = (2, 2).
        kwidth (float, optional): Width of the gaussian peak in the spectrum. Defaults to 12.0.
        do_truncation (bool, optional): Whether to truncate the spectrum. Defaults to False.
        n_cutoff (int, optional): Cutoff wave number for the spectrum. Defaults to None.
        do_mode_test (bool, optional): Whether to perform a single mode test. Defaults to False.

    Returns:
        None
    """                        

    EPS = 1e-30

    n_X = np.array(n_X)
    X_min = np.array(X_min)
    X_max = np.array(X_max)
    meshblock = np.array(meshblock)

    if folder[-1] != '/':
        folder += '/'

    h5name = folder + h5name

    N_HYDRO = 5
    one_D = 1 if np.all(n_X[1:] == 1) else 0

    # -------------------------------------------------
    # Background parameters
    # -------------------------------------------------

    B0_x, B0_y, B0_z = 1.0, 0.0, 0.0
    p0 = 0.5
    gamma = 5.0/3.0
    rho0_mean = 1.0

    Ly = X_max[1] - X_min[1]

    # Density with sinusoidal gradient
    Dnf = lambda X,Y,Z: rho0_mean*(1.0 + amp_rho*np.sin(2*np.pi*Y/Ly))

    UXf = lambda X,Y,Z: np.zeros_like(X)
    UYf = lambda X,Y,Z: np.zeros_like(X)
    UZf = lambda X,Y,Z: np.zeros_like(X)

    BXf = lambda X,Y,Z: B0_x*np.ones_like(X)
    BYf = lambda X,Y,Z: B0_y*np.ones_like(X)
    BZf = lambda X,Y,Z: B0_z*np.ones_like(X)

    # -------------------------------------------------
    # Generate grid
    # -------------------------------------------------

    X_grid, (dx,dy,dz) = helpers.generate_grid(X_min,X_max,n_X)

    Hy_grid, BXcc, BYcc, BZcc = helpers.setup_hydro_grid( n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)
    rho = Hy_grid[0]

    # -------------------------------------------------
    # IC SANITY CHECKS
    # -------------------------------------------------

    check_delta_rho(rho)
    check_alfven_speed(rho, BXcc)

    # -------------------------------------------------
    # Generate Alfven spectrum
    # -------------------------------------------------

    B0 = np.array([np.mean(BXcc), np.mean(BYcc), np.mean(BZcc)])

    if do_mode_test:

        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum, run_test=True)

    elif spectrum == 'gaussian':

        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            kpeak=kpeak, kwidth=kwidth,
            do_truncation=do_truncation,
            n_cutoff=n_cutoff)

    else:

        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            expo=expo, expo_prl=expo_prl,
            do_truncation=do_truncation,
            n_cutoff=n_cutoff)

    # -------------------------------------------------
    # Normalize perpendicular energy
    # -------------------------------------------------

    total_perp_energy = 0.5*np.mean(dB_x**2 + dB_y**2 + dB_z**2)

    if amplitude is not None:
        amp_u = amplitude/2
        target_energy = 0.5*amp_u**2
    else:
        target_energy = perp_energy

    norm = np.sqrt(target_energy/total_perp_energy)

    # -------------------------------------------------
    # Enforce RMHD Alfven relation
    # δu = δB / sqrt(ρ₀(y))
    # -------------------------------------------------

    du_x = dB_x / np.sqrt(rho + EPS)
    du_y = dB_y / np.sqrt(rho + EPS)
    du_z = dB_z / np.sqrt(rho + EPS)

    # -------------------------------------------------
    # Add perturbations
    # -------------------------------------------------

    Hy_grid[1] += rho*norm*du_x
    Hy_grid[2] += rho*norm*du_y
    Hy_grid[3] += rho*norm*du_z

    BXcc += norm*dB_x
    BYcc += norm*dB_y
    BZcc += norm*dB_z

    # -------------------------------------------------
    # Divergence check
    # -------------------------------------------------

    divB = (
        np.gradient(BXcc,dx,axis=2)
        + np.gradient(BYcc,dy,axis=1)
        + np.gradient(BZcc,dz,axis=0)
    )

    print("divB RMS =", np.sqrt(np.mean(divB**2)))

    # -------------------------------------------------
    # Elsasser check
    # -------------------------------------------------

    check_elsasser(Hy_grid, BXcc, BYcc, BZcc)

    # -------------------------------------------------
    # Build conserved energy
    # -------------------------------------------------

    rho = Hy_grid[0]

    vx = Hy_grid[1]/(rho+EPS)
    vy = Hy_grid[2]/(rho+EPS)
    vz = Hy_grid[3]/(rho+EPS)

    kin_e = 0.5*rho*(vx*vx + vy*vy + vz*vz)
    mag_e = 0.5*(BXcc**2 + BYcc**2 + BZcc**2)
    gas_e = p0/(gamma-1.0)

    E = kin_e + mag_e + gas_e

    Hy_grid[4] = E

    # -------------------------------------------------
    # Sound speed for athinput
    # -------------------------------------------------

    B2 = BXcc**2 + BYcc**2 + BZcc**2
    mean_rho_on_B2 = box_avg(rho/B2)

    iso_sound_speed = np.sqrt(0.5*beta/mean_rho_on_B2)

    ath_copy = helpers.edit_athinput(
        athinput, folder,
        n_X, X_min, X_max,
        meshblock,
        h5name,
        time_lim, dt,
        iso_sound_speed,
        expand, exp_rate)

    # -------------------------------------------------
    # Meshblocks
    # -------------------------------------------------

    n_blocks, blocks = helpers.make_meshblocks(
        folder, ath_copy,
        n_X, meshblock, one_D)

    helpers.remove_prev_h5file(h5name)

    helpers.calc_and_save_B(
        BXcc, BYcc, BZcc,
        h5name, n_X,
        X_min, X_max,
        meshblock,
        n_blocks, blocks,
        dx, dy, dz)

    print("Magnetic Saved Successfully")

    helpers.save_hydro_grid(
        h5name,
        Hy_grid,
        N_HYDRO,
        n_blocks,
        blocks,
        meshblock,
        remove_h5=0)

    print("Hydro Saved Successfully")
