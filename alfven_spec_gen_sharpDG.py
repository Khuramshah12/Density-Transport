# --- IMPORTS --- #
import numpy as np

import generate_spectrum as genspec
import helper_functions as helpers
from ic_checks import check_delta_rho, check_elsasser, check_alfven_speed


# =============================================================================
#  UTILITIES
# =============================================================================

def box_avg(x, reshape=0):
    """Average over the trailing three (grid) axes."""
    len_shape = len(x.shape)
    if len_shape < 3:
        return x
    axes = tuple(np.arange(len_shape - 3, len_shape))
    avg = np.mean(x, axis=axes)
    if reshape:
        shape = avg.shape
        avg = avg.reshape(*shape, 1, 1, 1)
    return avg


# =============================================================================
#ics.create_athena_alfvenspec('/nesi/nobackup/uoo02637/khurram/density_transport/500_k15/Sharp_grad/cubic_rho0_5kp15B01A02/', 'ics.h5', [500, 500, 500], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [100, 100, 125], '/nesi/nobackup/uoo02637/khurram/density_transport/500_k15/Sharp_grad/cubic_rho0_5kp15B01A02/athinput.from_array', time_lim=100.0, dt=1, expand=False, exp_rate=0.0, beta=0.1, perp_energy=0.5, amplitude=0.2, spectrum='gaussian', expo=-5/3, expo_prl=-2., kpeak=(1,5), kwidth=12.0, amp_rho = 0.5, alpha_steepness=10.0, do_truncation=False, n_cutoff=None, do_mode_test=False)

#ics.create_athena_alfvenspec('/nesi/nobackup/uoo02637/khurram/density_transport/500_k15/Sharp_grad/rho0_9_kp15B01A02/', 'ics.h5', [250, 500, 500], [0.0, 0.0, 0.0], [5.0, 1.0, 1.0], [50, 125, 100], '/nesi/nobackup/uoo02637/khurram/density_transport/500_k15/Sharp_grad/rho0_9_kp15B01A02/athinput.from_array', time_lim=100.0, dt=1, expand=False, exp_rate=0.0, beta=0.1, perp_energy=0.5, amplitude=0.2, spectrum='gaussian', expo=-5/3, expo_prl=-2., kpeak=(1,5), kwidth=12.0, amp_rho = 0.9, alpha_steepness=10.0, do_truncation=False, n_cutoff=None, do_mode_test=False)
# =============================================================================
#  PHYSICS HELPERS  (tanh profile, scale matching, critical balance)
# =============================================================================

def kperp_max_tanh(amp_rho, alpha_steepness, kappa_y, Ly):
    """Maximum local perpendicular inverse scale length of the tanh density
    profile

        rho(y) = rho_bar * [1 + A_rho * tanh(alpha * sin(Ky*y)) / tanh(alpha)].

    The local gradient peaks at the interfaces (sin(Ky*y) = 0), where
    sech^2(0) = 1 and rho ~ rho_bar, giving

        K_perp_max = A_rho * alpha * Ky / tanh(alpha).

    For alpha >~ 3 this reduces to the user-quoted approximation
    K_perp_max ~ A_rho * alpha * Ky.
    """
    Ky = 2.0 * np.pi * kappa_y / Ly
    return amp_rho * alpha_steepness * Ky / np.tanh(alpha_steepness)


def gradient_thickness(amp_rho, alpha_steepness, kappa_y, Ly):
    """Characteristic physical thickness of the tanh interface,
    delta_y = 1 / K_perp_max.  This is the length scale that the
    perpendicular grid must resolve."""
    return 1.0 / kperp_max_tanh(amp_rho, alpha_steepness, kappa_y, Ly)


def required_zrms_for_chi_unity(kappa_par, kappa_perp_eff, Lx, Ly, vA=1.0):
    """Solve chi_A = z+_rms * k_perp / (v_A * k_par) = 1 for z+_rms,
    given integer cycles kappa_par (in x) and effective integer-cycles
    kappa_perp_eff (in y) selected to match the gradient scale."""
    k_par  = 2.0 * np.pi * kappa_par / Lx
    k_perp = 2.0 * np.pi * kappa_perp_eff / Ly
    return vA * k_par / k_perp


def required_Lx_for_chi_unity(kappa_par, kappa_perp_eff, Ly,
                              zrms_target, vA=1.0):
    """Conversely, if the injection amplitude z+_rms is fixed, return the
    parallel box length L_x that yields chi_A = 1."""
    k_perp = 2.0 * np.pi * kappa_perp_eff / Ly
    return 2.0 * np.pi * kappa_par * vA / (zrms_target * k_perp)


def sound_speed_from_beta(beta, vA_mean, gamma=5.0/3.0):
    """c_s = v_A * sqrt(beta * gamma / 2).  Returned in code units."""
    return vA_mean * np.sqrt(beta * gamma / 2.0)


def select_kappa_perp_eff(K_perp_max_phys, Ly, prefer='nearest_int'):
    """Convert the physical perpendicular peak K_perp_max to integer cycles
    on a periodic box of size Ly.  Spectrum modes live on the integer
    lattice, so we must snap to a nearby integer.  'prefer' selects
    rounding behaviour ('floor', 'ceil', 'nearest_int')."""
    k_eff_real = K_perp_max_phys * Ly / (2.0 * np.pi)
    if   prefer == 'floor':        return max(1, int(np.floor(k_eff_real)))
    elif prefer == 'ceil':         return max(1, int(np.ceil(k_eff_real)))
    else:                          return max(1, int(np.round(k_eff_real)))


def resolution_warning(amp_rho, alpha_steepness, kappa_y, Ly, Ny,
                       min_cells_per_grad=6):
    """Verify that at least 'min_cells_per_grad' grid cells span the tanh
    interface thickness.  Returns (ok, n_cells_per_grad, dy, delta_y)."""
    delta_y = gradient_thickness(amp_rho, alpha_steepness, kappa_y, Ly)
    dy      = Ly / Ny
    n_cells = delta_y / dy
    return (n_cells >= min_cells_per_grad), n_cells, dy, delta_y


# =============================================================================
#  MAIN CONSTRUCTOR
# =============================================================================

def create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock,
                             athinput,
                             time_lim=1, dt=0.2,
                             expand=False, exp_rate=0., beta=1.0,
                             perp_energy=0.5, amplitude=None,
                             spectrum='isotropic',
                             expo=-5/3, expo_prl=-2.,
                             kpeak=(2, 2), kwidth=12.0,
                             # --- tanh density profile params ---
                             amp_rho=0.0,
                             alpha_steepness=5.0,
                             kappa_y=1,
                             # --- spectrum tuning ---
                             auto_match_kperp=True,
                             kappa_par_for_chi=1,
                             do_truncation=False, n_cutoff=None,
                             do_mode_test=False,
                             # --- resolution check ---
                             min_cells_per_grad=6):
    """
    Generate an Athena++ initial-condition snapshot with a sharp tanh
    density interface and a critically balanced Alfvenic spectrum whose
    perpendicular peak matches the gradient scale.

    Key new parameters
    ------------------
    amp_rho            : density contrast A_rho (typical 0.1 - 0.5)
    alpha_steepness    : tanh steepness alpha (typical 3 - 10)
    kappa_y            : integer cycles of the density modulation in y
    auto_match_kperp   : if True, the injected spectrum's perpendicular
                         peak is set to the integer cycles closest to
                         K_perp_max of the density profile
    kappa_par_for_chi  : parallel integer cycles assumed in the
                         critical-balance diagnostic (chi_A target)
    min_cells_per_grad : minimum number of grid cells required to resolve
                         the tanh interface; a warning is printed below it

    Diagnostics printed at construction
    -----------------------------------
    * Mean and extremal density, Alfven speed
    * Sound speed c_s = v_A * sqrt(beta*gamma/2)
    * K_perp_max of the tanh profile, and the integer kappa_perp chosen
    * Required z+_rms for chi_A = 1 at the current box
    * Required L_x for chi_A = 1 if the injection amplitude were held at
      the value implied by the user-supplied 'amplitude' or 'perp_energy'
    * Resolution check across the tanh interface
    """

    EPS = 1e-30
    GAMMA = 5.0 / 3.0
    p0    = 0.5
    rho0_mean = 1.0

    n_X       = np.array(n_X)
    X_min     = np.array(X_min)
    X_max     = np.array(X_max)
    meshblock = np.array(meshblock)

    if folder[-1] != '/':
        folder += '/'
    h5name = folder + h5name

    N_HYDRO = 5
    one_D   = 1 if np.all(n_X[1:] == 1) else 0

    Lx = X_max[0] - X_min[0]
    Ly = X_max[1] - X_min[1]
    Lz = X_max[2] - X_min[2]

    B0_x, B0_y, B0_z = 1.0, 0.0, 0.0

    # -------------------------------------------------------------------
    #  1.  Tanh density profile
    # -------------------------------------------------------------------
    Ky = 2.0 * np.pi * kappa_y / Ly

    def Dnf(X, Y, Z):
        # rho(y) = rho_bar * (1 + A_rho * tanh(alpha sin(Ky y)) / tanh(alpha))
        if amp_rho == 0.0 or alpha_steepness == 0.0:
            return rho0_mean * np.ones_like(X)
        return rho0_mean * (
            1.0 + amp_rho * np.tanh(alpha_steepness * np.sin(Ky * Y))
                          / np.tanh(alpha_steepness)
        )

    UXf = lambda X, Y, Z: np.zeros_like(X)
    UYf = lambda X, Y, Z: np.zeros_like(X)
    UZf = lambda X, Y, Z: np.zeros_like(X)
    BXf = lambda X, Y, Z: B0_x * np.ones_like(X)
    BYf = lambda X, Y, Z: B0_y * np.ones_like(X)
    BZf = lambda X, Y, Z: B0_z * np.ones_like(X)

    # -------------------------------------------------------------------
    #  2.  Build grid and hydro arrays
    # -------------------------------------------------------------------
    X_grid, (dx, dy, dz) = helpers.generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = helpers.setup_hydro_grid(
        n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf
    )
    rho = Hy_grid[0]

    Bmag    = np.sqrt(BXcc**2 + BYcc**2 + BZcc**2)
    vA_grid = Bmag / np.sqrt(rho + EPS)
    vA_mean = float(np.mean(vA_grid))
    cs      = sound_speed_from_beta(beta, vA_mean, gamma=GAMMA)

    # -------------------------------------------------------------------
    #  3.  Scale matching: choose kappa_perp to align with the gradient
    # -------------------------------------------------------------------
    K_perp_max = kperp_max_tanh(amp_rho, alpha_steepness, kappa_y, Ly) \
                 if amp_rho > 0 else 0.0
    kappa_perp_eff = (
        select_kappa_perp_eff(K_perp_max, Ly, prefer='nearest_int')
        if (auto_match_kperp and amp_rho > 0)
        else (kpeak[1] if isinstance(kpeak, (tuple, list)) else int(kpeak))
    )

    # Replace kpeak[1] with the matched value when auto-matching is on.
    if auto_match_kperp and amp_rho > 0:
        kpeak_used = (kpeak[0] if isinstance(kpeak, (tuple, list)) else 1,
                      kappa_perp_eff)
    else:
        kpeak_used = kpeak

    # -------------------------------------------------------------------
    #  4.  Critical-balance diagnostics
    # -------------------------------------------------------------------
    zrms_target = required_zrms_for_chi_unity(
        kappa_par_for_chi, kappa_perp_eff, Lx, Ly, vA=vA_mean
    ) if kappa_perp_eff > 0 else np.nan

    # If user provided 'amplitude', infer its z+_rms (z+_rms ~ amplitude
    # under the existing perp-isotropy convention) and compute the L_x
    # that would yield chi_A=1 at that injection level.
    if amplitude is not None:
        zrms_inferred = amplitude
    else:
        zrms_inferred = np.sqrt(2.0 * perp_energy)
    Lx_for_chi_unity = required_Lx_for_chi_unity(
        kappa_par_for_chi, kappa_perp_eff, Ly,
        zrms_target=zrms_inferred, vA=vA_mean
    ) if kappa_perp_eff > 0 else np.nan

    # -------------------------------------------------------------------
    #  5.  Resolution warning
    # -------------------------------------------------------------------
    if amp_rho > 0:
        ok, ncells, dy_grid, delta_y = resolution_warning(
            amp_rho, alpha_steepness, kappa_y, Ly, n_X[1],
            min_cells_per_grad=min_cells_per_grad
        )
    else:
        ok, ncells, dy_grid, delta_y = True, np.inf, Ly / n_X[1], np.inf

    # -------------------------------------------------------------------
    #  6.  Setup report
    # -------------------------------------------------------------------
    print('\n' + '=' * 64)
    print('  RMHD setup: tanh interface + scale-matched spectrum')
    print('=' * 64)
    print(f'  Box:                Lx={Lx:.3g}  Ly={Ly:.3g}  Lz={Lz:.3g}')
    print(f'  Grid:               Nx={n_X[0]}  Ny={n_X[1]}  Nz={n_X[2]}')
    print(f'  Plasma beta:        {beta:.3g}')
    print(f'  Mean v_A:           {vA_mean:.4f}')
    print(f'  Sound speed c_s:    {cs:.4f}    '
          f'(c_s = v_A*sqrt(beta*gamma/2), gamma={GAMMA:.3f})')
    print(f'  Density min/max:    {rho.min():.4f} / {rho.max():.4f}')
    print(f'  A_rho:              {amp_rho:.3g}')
    print(f'  alpha_steepness:    {alpha_steepness:.3g}')
    print(f'  kappa_y (cycles):   {kappa_y}')
    print(f'  K_perp_max (rad/L): {K_perp_max:.4f}')
    print(f'                      = A_rho * alpha * Ky / tanh(alpha)')
    print(f'  kappa_perp_eff:     {kappa_perp_eff}  '
          f'(integer cycles matched to gradient)')
    print()
    print(f'  Critical-balance diagnostic (target chi_A = 1):')
    print(f'    assume kappa_par = {kappa_par_for_chi}')
    print(f'    required z+_rms  = {zrms_target:.4f}   '
          f'(at current Lx={Lx:.3g})')
    print(f'    OR required Lx   = {Lx_for_chi_unity:.4f}   '
          f'(at current z+_rms={zrms_inferred:.4f})')
    print()
    if amp_rho > 0:
        msg = 'OK' if ok else 'WARNING'
        print(f'  Resolution check ({msg}):')
        print(f'    interface thickness delta_y = {delta_y:.4g}')
        print(f'    grid spacing       dy       = {dy_grid:.4g}')
        print(f'    cells per gradient          = {ncells:.2f}  '
              f'(min required: {min_cells_per_grad})')
        if not ok:
            print(f'    >>> Ny is too coarse for alpha={alpha_steepness}; '
                  f'consider Ny >= {int(np.ceil(min_cells_per_grad/delta_y*Ly))}')
    print('=' * 64 + '\n')

    # -------------------------------------------------------------------
    #  7.  IC sanity checks
    # -------------------------------------------------------------------
    check_delta_rho(rho)
    check_alfven_speed(rho, BXcc)

    # -------------------------------------------------------------------
    #  8.  Generate Alfven spectrum (perp peak matched to gradient)
    # -------------------------------------------------------------------
    B0 = np.array([np.mean(BXcc), np.mean(BYcc), np.mean(BZcc)])

    if do_mode_test:
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum, run_test=True
        )
    elif spectrum == 'gaussian':
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            kpeak=kpeak_used, kwidth=kwidth,
            do_truncation=do_truncation, n_cutoff=n_cutoff
        )
    else:
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            expo=expo, expo_prl=expo_prl,
            do_truncation=do_truncation, n_cutoff=n_cutoff
        )

    # -------------------------------------------------------------------
    #  9.  Normalise perpendicular energy
    # -------------------------------------------------------------------
    total_perp_energy = 0.5 * np.mean(dB_x**2 + dB_y**2 + dB_z**2)
    if amplitude is not None:
        amp_u = amplitude / 2.0
        target_energy = 0.5 * amp_u**2
    else:
        target_energy = perp_energy
    norm = np.sqrt(target_energy / total_perp_energy)

    # RMHD Alfven relation: du = dB / sqrt(rho)
    du_x = dB_x / np.sqrt(rho + EPS)
    du_y = dB_y / np.sqrt(rho + EPS)
    du_z = dB_z / np.sqrt(rho + EPS)

    Hy_grid[1] += rho * norm * du_x
    Hy_grid[2] += rho * norm * du_y
    Hy_grid[3] += rho * norm * du_z
    BXcc       += norm * dB_x
    BYcc       += norm * dB_y
    BZcc       += norm * dB_z

    # divergence and Elsasser checks
    divB = (np.gradient(BXcc, dx, axis=2)
            + np.gradient(BYcc, dy, axis=1)
            + np.gradient(BZcc, dz, axis=0))
    print('divB RMS =', np.sqrt(np.mean(divB**2)))
    check_elsasser(Hy_grid, BXcc, BYcc, BZcc)

    # -------------------------------------------------------------------
    # 10.  Build conserved energy
    # -------------------------------------------------------------------
    rho = Hy_grid[0]
    vx, vy, vz = (Hy_grid[1] / (rho + EPS),
                  Hy_grid[2] / (rho + EPS),
                  Hy_grid[3] / (rho + EPS))
    kin_e = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
    mag_e = 0.5 * (BXcc**2 + BYcc**2 + BZcc**2)
    gas_e = p0 / (GAMMA - 1.0)
    Hy_grid[4] = kin_e + mag_e + gas_e

    # -------------------------------------------------------------------
    # 11.  Isothermal sound speed for athinput
    # -------------------------------------------------------------------
    B2 = BXcc**2 + BYcc**2 + BZcc**2
    mean_rho_on_B2 = box_avg(rho / B2)
    iso_sound_speed = np.sqrt(0.5 * beta / mean_rho_on_B2)

    ath_copy = helpers.edit_athinput(
        athinput, folder, n_X, X_min, X_max, meshblock,
        h5name, time_lim, dt, iso_sound_speed, expand, exp_rate
    )

    # -------------------------------------------------------------------
    # 12.  Meshblocks and HDF5 output
    # -------------------------------------------------------------------
    n_blocks, blocks = helpers.make_meshblocks(folder, ath_copy,
                                               n_X, meshblock, one_D)
    helpers.remove_prev_h5file(h5name)
    helpers.calc_and_save_B(BXcc, BYcc, BZcc,
                            h5name, n_X, X_min, X_max,
                            meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    helpers.save_hydro_grid(h5name, Hy_grid, N_HYDRO,
                            n_blocks, blocks, meshblock, remove_h5=0)
    print('Hydro Saved Successfully')

    # Return the diagnostic summary so a calling driver can log it
    return dict(
        vA_mean=vA_mean, cs=cs,
        K_perp_max=K_perp_max,
        kappa_perp_eff=kappa_perp_eff,
        zrms_target_chi1=zrms_target,
        Lx_for_chi1_at_inferred_zrms=Lx_for_chi_unity,
        resolution_ok=ok,
        cells_per_gradient=ncells,
        interface_thickness=delta_y,
    )
