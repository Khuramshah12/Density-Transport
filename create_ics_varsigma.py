# =====================================================================
#  create_ics_varsigma.py
#  Athena++ MHD initial conditions with SPATIALLY VARYING cross-helicity
#
#  Physics (velocity-only construction; delta-B left untouched):
#     b(y)      = delta_B / sqrt(rho)              (Alfven-unit magnetic field)
#     sigma_c(y)= tanh map of background density   (open -> imbalanced)
#     delta_v   = sigma_c * b  +  sqrt(1-sigma_c^2) * v_indep
#       => sigma_c = c exactly, zero residual energy (z+/z- decorrelated)
#     v_indep normalized to local <b^2>(y)  => E_v(y) = E_b(y), r_A = 1
#     optional solenoidal projection of total delta_v suppresses
#       envelope-induced div(v); delta_B is never masked so div(B)=0 exactly.
#
#  Author: Khurram
# =====================================================================

import numpy as np

import generate_spectrum as genspec
import helper_functions as helpers
from ic_checks import check_delta_rho, check_elsasser, check_alfven_speed


# ---------------------------------------------------------------------
#  box average (unchanged from original)
# ---------------------------------------------------------------------
def box_avg(x, reshape=0):
    """Average over the trailing 3 (spatial) axes of a box quantity.

    If `x` has fewer than 3 dims it is returned unchanged (already reduced
    or not a box field). With `reshape=1` the three averaged axes are added
    back as length-1 dims so the result broadcasts against the full grid.
    """
    len_shape = len(x.shape)
    if len_shape < 3:
        return x
    axes = tuple(np.arange(len_shape - 3, len_shape))
    avg = np.mean(x, axis=axes)
    if reshape:
        shape = avg.shape
        avg = avg.reshape(*shape, 1, 1, 1)
    return avg


# =====================================================================
#  CROSS-HELICITY ENGINEERING HELPERS
# =====================================================================

def sigma_c_of_rho(rho, sigma_hi=0.9, sigma_lo=0.0, rho_t=1.0, width=0.05,
                   use_log=False):
    """Target cross-helicity profile as a tanh function of background density.

        sigma_c(rho) = sigma_lo
                     + (sigma_hi - sigma_lo)/2 * (1 - tanh((coord - coord_t)/width))

    where coord = rho (or ln rho if `use_log`). Low density -> sigma_hi
    (open/imbalanced); high density -> sigma_lo (closed/balanced). The tanh
    gives flat plateaus that actually reach the targets, a tunable transition
    `width`, and smooth derivatives (so the induced div(v) stays bounded).
    Output is clipped to [0, 1] so the later sqrt(1 - sigma_c^2) is always real.

    Parameters
    ----------
    rho : ndarray              background density field (any shape).
    sigma_hi, sigma_lo : float low-/high-density plateau values of sigma_c.
    rho_t : float              transition density (centre of the tanh).
    width : float              transition width in the chosen coordinate.
    use_log : bool             use ln(rho) as the transition coordinate.

    Returns
    -------
    ndarray (same shape as rho) of target sigma_c in [0, 1].
    """
    coord = np.log(rho) if use_log else rho
    coord_t = np.log(rho_t) if use_log else rho_t
    sc = sigma_lo + 0.5 * (sigma_hi - sigma_lo) * (1.0 - np.tanh((coord - coord_t) / width))
    return np.clip(sc, 0.0, 1.0)


def project_solenoidal(vx, vy, vz, dx, dy, dz):
    """Remove the compressive part of a vector field in Fourier space.

    Applies the transverse (Leray) projector  v_sol = (I - k_hat k_hat) . v,
    component-wise:
        v_i(k) <- v_i(k) - k_i (k . v) / |k|^2 .
    The result satisfies div(v_sol) = 0 to round-off. Used to suppress the
    div(v) introduced when the solenoidal fields b and v_indep are multiplied
    by the spatially varying envelopes sigma_c(y) and sqrt(1 - sigma_c^2)(y).

    Grid spacings dx, dy, dz map to FFT axes (x=2, y=1, z=0) in Athena order.
    """
    nz, ny, nx = vx.shape
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0   # protect the k=0 (mean) mode from division by zero

    fx = np.fft.fftn(vx)
    fy = np.fft.fftn(vy)
    fz = np.fft.fftn(vz)

    kdotf = (KX * fx + KY * fy + KZ * fz) / k2
    fx -= kdotf * KX
    fy -= kdotf * KY
    fz -= kdotf * KZ

    return (np.real(np.fft.ifftn(fx)),
            np.real(np.fft.ifftn(fy)),
            np.real(np.fft.ifftn(fz)))


def yprofile_rms(field):
    """RMS of a box field over the perpendicular (x, z) planes at each y.

    `field` has Athena shape (nz, ny, nx); averaging axes (0, 2) leaves a
    1D profile of length ny. Used to build <b^2>(y) and <v_indep^2>(y) so the
    seed can be matched to the local magnetic-velocity energy.
    """
    return np.sqrt(np.mean(field**2, axis=(0, 2)))


def crosshel_diagnostics(dv_x, dv_y, dv_z, dB_x, dB_y, dB_z, rho, p0, B0_sq, BXcc, BYcc, BZcc, beta,
                         sigma_c_target=None, dx=1.0, dy=1.0, dz=1.0,
                         EPS=1e-30):
    """Verify the engineered IC before it is written to disk.

    Builds z+- = delta_v +- delta_B/sqrt(rho), reduces to y-profiles via
    <.>_xz, and reports:
        sigma_c(y) = (E+ - E-)/(E+ + E-)          measured cross-helicity
        r_A(y)     = E_v / E_b                     Alfven ratio (~1 by design)
        z+_rms(y)  = sqrt(2 E+)                    carries physical sqrt(1+sigma_c)/sqrt(rho)
        div(v) RMS                                 residual compressive injection

    If `sigma_c_target` is given, prints the worst-case |measured - target| so
    a large solenoidal-projection drift is visible immediately.
    Returns the measured y-profiles as a dict.
    """
    inv_sqrt_rho = 1.0 / np.sqrt(rho + EPS)
    zpx = dv_x + dB_x * inv_sqrt_rho
    zpy = dv_y + dB_y * inv_sqrt_rho
    zpz = dv_z + dB_z * inv_sqrt_rho
    zmx = dv_x - dB_x * inv_sqrt_rho
    zmy = dv_y - dB_y * inv_sqrt_rho
    zmz = dv_z - dB_z * inv_sqrt_rho

    def prof(*comps):
        return np.mean(sum(c**2 for c in comps), axis=(0, 2))   # <.>_xz -> (ny,)

    Ep = 0.5 * prof(zpx, zpy, zpz)
    Em = 0.5 * prof(zmx, zmy, zmz)
    Ev = 0.5 * prof(dv_x, dv_y, dv_z)
    Eb = 0.5 * prof(dB_x * inv_sqrt_rho, dB_y * inv_sqrt_rho, dB_z * inv_sqrt_rho)

    sigma_c_meas = (Ep - Em) / (Ep + Em + EPS)
    r_A = Ev / (Eb + EPS)
    zp_rms = np.sqrt(2.0 * Ep)

    beta_eff_B0 = 2.0 * p0 / B0_sq
    beta_eff_full = 2.0 * p0 / np.mean(BXcc**2 + BYcc**2 + BZcc**2)
    print(f"  beta(B0^2)      = {beta_eff_B0:.4g}   (target = {beta})")
    print(f"  beta(<B^2>)     = {beta_eff_full:.4g}")


    divv = (np.gradient(dv_x, dx, axis=2)
            + np.gradient(dv_y, dy, axis=1)
            + np.gradient(dv_z, dz, axis=0))

    print("---- cross-helicity IC diagnostics ----")
    print(f"  sigma_c(y): measured [{sigma_c_meas.min():.3f}, {sigma_c_meas.max():.3f}]")
    if sigma_c_target is not None:
        tgt = np.mean(sigma_c_target, axis=(0, 2))
        print(f"              target   [{tgt.min():.3f}, {tgt.max():.3f}]  "
              f"max|delta| = {np.max(np.abs(sigma_c_meas - tgt)):.3f}")
    print(f"  r_A(y):     [{r_A.min():.3f}, {r_A.max():.3f}]   (~1 expected)")
    print(f"  z+_rms(y):  [{zp_rms.min():.4f}, {zp_rms.max():.4f}]")
    print(f"  div(v) RMS = {np.sqrt(np.mean(divv**2)):.3e}")
    print("----------------------------------------")
    return dict(y_sigma_c=sigma_c_meas, y_r_A=r_A, y_zp_rms=zp_rms)


def create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock, athinput,
                             time_lim=1, dt=0.2, expand=False, exp_rate=0., beta=1.0,
                             perp_energy=0.5, amplitude=None,
                             spectrum='isotropic', expo=-5/3, expo_prl=-2.,
                             kpeak=(2, 2), kwidth=12.0,
                             amp_rho=0.0,
                             rho_alpha=4.0,
                             do_truncation=False, n_cutoff=None,
                             do_mode_test=False,
                             # --- spatially varying cross-helicity controls ---
                             sigma_hi=0.9, sigma_lo=0.0, rho_t=1.0,
                             sigma_width=0.05, sigma_use_log=False,
                             do_solenoidal_v=True, seed_offset=12345,
                             use_vectorized_seed=True):
    """Generate Athena++ IC HDF5 with a spatially varying cross-helicity.

    The magnetic perturbation delta_B is built and amplitude-normalized once,
    then frozen (never masked) so the staggered-mesh curl write keeps div(B)=0.
    The velocity perturbation is engineered to hit a target sigma_c(rho):

        delta_v = sigma_c(y) * (delta_B/sqrt(rho))
                  + sqrt(1 - sigma_c(y)^2) * v_indep ,

    with v_indep an independent solenoidal seed scaled to the local
    magnetic-velocity energy so that r_A = E_v/E_b = 1 at every y.

    New parameters
    --------------
    sigma_hi, sigma_lo : float   low-/high-density sigma_c plateaus.
    rho_t : float                tanh transition density.
    sigma_width : float          tanh transition width (>= a few cells of the
                                 rho profile to keep induced div(v) small).
    sigma_use_log : bool         use ln(rho) as the transition coordinate.
    do_solenoidal_v : bool       project total delta_v solenoidal at the end.
    seed_offset : int            RNG offset guaranteeing the seed realization
                                 differs from the delta_B realization.
    use_vectorized_seed : bool   use genspec.generate_alfven_spectrum_vectorized
                                 for the seed (fast); set False to force the
                                 per-mode genspec.generate_alfven_spectrum.
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
    # Gas pressure from target beta: beta = 2 P_th / B0^2  (mu_0 = 1 in code units)
    B0_sq = B0_x**2 + B0_y**2 + B0_z**2
    p0 = 0.5 * beta * B0_sq
    print(f"  target beta = {beta}   ->   p0 = {p0:.6g}   (B0^2 = {B0_sq:.4g})")
    
    #p0 = 0.5
    gamma = 5.0 / 3.0
    rho0_mean = 1.0

    Ly = X_max[1] - X_min[1]

    Ky = 2 * np.pi / Ly

    # Square-like density profile via tanh-sharpened sine.
    #   rho(y) = rho_bar (1 + A_rho * tanh(alpha sin(Ky y)) / tanh(alpha))
    # alpha -> 0 recovers the pure sine; alpha >~ 3 gives near-square plateaus.
    # The /tanh(alpha) keeps the peak amplitude exactly A_rho, so rho spans
    # [1 - A_rho, 1 + A_rho] regardless of alpha.
    def Dnf(X, Y, Z):
        if rho_alpha <= 1e-8:                      # graceful sine limit
            shape = np.sin(Ky * Y)
        else:
            shape = np.tanh(rho_alpha * np.sin(Ky * Y)) / np.tanh(rho_alpha)
        return rho0_mean * (1.0 + amp_rho * shape)

    # Density with sinusoidal gradient
    #Dnf = lambda X, Y, Z: rho0_mean * (1.0 + amp_rho * np.sin(2 * np.pi * Y / Ly))
    UXf = lambda X, Y, Z: np.zeros_like(X)
    UYf = lambda X, Y, Z: np.zeros_like(X)
    UZf = lambda X, Y, Z: np.zeros_like(X)
    BXf = lambda X, Y, Z: B0_x * np.ones_like(X)
    BYf = lambda X, Y, Z: B0_y * np.ones_like(X)
    BZf = lambda X, Y, Z: B0_z * np.ones_like(X)

    # -------------------------------------------------
    # Generate grid + place background fields
    # -------------------------------------------------
    X_grid, (dx, dy, dz) = helpers.generate_grid(X_min, X_max, n_X)

    Hy_grid, BXcc, BYcc, BZcc = helpers.setup_hydro_grid(
        n_X, X_grid, N_HYDRO,
        Dnf, UXf, UYf, UZf,
        BXf, BYf, BZf)

    rho = Hy_grid[0]

    # -------------------------------------------------
    # IC SANITY CHECKS (background)
    # -------------------------------------------------
    check_delta_rho(rho)
    check_alfven_speed(rho, BXcc)

    # -------------------------------------------------
    # Generate the magnetic Alfven spectrum (delta_B)
    # -------------------------------------------------
    B0 = np.array([np.mean(BXcc), np.mean(BYcc), np.mean(BZcc)])

    if do_mode_test:
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum, run_test=True)
    elif spectrum == 'gaussian':
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            kpeak=kpeak, kwidth=kwidth,
            do_truncation=do_truncation, n_cutoff=n_cutoff)
    else:
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            expo=expo, expo_prl=expo_prl,
            do_truncation=do_truncation, n_cutoff=n_cutoff)

    # -------------------------------------------------
    # Amplitude normalization -> applied to delta_B FIRST
    #   (so the velocity, built from delta_B/sqrt(rho), inherits it)
    # -------------------------------------------------
    total_perp_energy = 0.5 * np.mean(dB_x**2 + dB_y**2 + dB_z**2)
    if amplitude is not None:
        amp_u = amplitude / 2
        target_energy = 0.5 * amp_u**2
    else:
        target_energy = perp_energy
    norm = np.sqrt(target_energy / (total_perp_energy + EPS))

    dB_x *= norm
    dB_y *= norm
    dB_z *= norm

    # -------------------------------------------------
    # Correlated magnetic-velocity field  b = delta_B / sqrt(rho)
    #   (this is your original RMHD Alfven relation)
    # -------------------------------------------------
    inv_sqrt_rho = 1.0 / np.sqrt(rho + EPS)
    b_x = dB_x * inv_sqrt_rho
    b_y = dB_y * inv_sqrt_rho
    b_z = dB_z * inv_sqrt_rho

    # -------------------------------------------------
    # Independent solenoidal velocity seed  v_indep
    #   Fresh RNG realization (independent of delta_B), then projected
    #   divergence-free. Vectorized generator used for speed on large grids.
    # -------------------------------------------------
    np.random.seed(seed_offset)
    if use_vectorized_seed:
        vi_x, vi_y, vi_z = genspec.generate_alfven_spectrum_vectorized(
            n_X, X_min, X_max, B0, spectrum=spectrum,
            expo=expo, expo_prl=expo_prl, kpeak=kpeak, kwidth=kwidth,
            do_truncation=do_truncation, n_cutoff=n_cutoff)
    else:
        vi_x, vi_y, vi_z = genspec.generate_alfven_spectrum(
            n_X, X_min, X_max, B0, spectrum,
            kpeak=kpeak, kwidth=kwidth,
            do_truncation=do_truncation, n_cutoff=n_cutoff)

    vi_x, vi_y, vi_z = project_solenoidal(vi_x, vi_y, vi_z, dx, dy, dz)

    # -------------------------------------------------
    # Target sigma_c(rho) and the two mixing weights
    #   c = sigma_c , s = sqrt(1 - sigma_c^2)  -> sigma_c = c, zero residual E
    # -------------------------------------------------
    sigma_c_3d = sigma_c_of_rho(rho, sigma_hi=sigma_hi, sigma_lo=sigma_lo,
                                rho_t=rho_t, width=sigma_width,
                                use_log=sigma_use_log)
    c_w = sigma_c_3d
    s_w = np.sqrt(np.clip(1.0 - sigma_c_3d**2, 0.0, 1.0))

    # -------------------------------------------------
    # Normalize seed to local <b^2>(y) so E_v(y) = E_b(y)  (r_A = 1)
    # -------------------------------------------------
    b2_prof = (yprofile_rms(b_x)**2 + yprofile_rms(b_y)**2 + yprofile_rms(b_z)**2)
    vi2_prof = (yprofile_rms(vi_x)**2 + yprofile_rms(vi_y)**2 + yprofile_rms(vi_z)**2)
    seed_scale_y = np.sqrt(b2_prof / (vi2_prof + EPS))          # (ny,)
    seed_scale = seed_scale_y[None, :, None]                    # broadcast (z, y, x)
    vi_x = vi_x * seed_scale
    vi_y = vi_y * seed_scale
    vi_z = vi_z * seed_scale

    # -------------------------------------------------
    # Superpose:  delta_v = c(y)*b + s(y)*v_indep
    # -------------------------------------------------
    dv_x = c_w * b_x + s_w * vi_x
    dv_y = c_w * b_y + s_w * vi_y
    dv_z = c_w * b_z + s_w * vi_z

    # -------------------------------------------------
    # Optional: project total delta_v solenoidal (suppress envelope div(v))
    # -------------------------------------------------
    if do_solenoidal_v:
        dv_x, dv_y, dv_z = project_solenoidal(dv_x, dv_y, dv_z, dx, dy, dz)

    # -------------------------------------------------
    # Diagnostics (verify sigma_c(y), r_A(y), z+_rms(y), div(v))
    # -------------------------------------------------
    crosshel_diagnostics(dv_x, dv_y, dv_z, dB_x, dB_y, dB_z, rho, p0, B0_sq, BXcc, BYcc, BZcc, beta,
                         sigma_c_target=sigma_c_3d, dx=dx, dy=dy, dz=dz, EPS=EPS)

    # -------------------------------------------------
    # Add perturbations
    #   velocity as MOMENTUM (conserved form);  delta_B already normalized
    # -------------------------------------------------
    Hy_grid[1] += rho * dv_x
    Hy_grid[2] += rho * dv_y
    Hy_grid[3] += rho * dv_z

    BXcc += dB_x
    BYcc += dB_y
    BZcc += dB_z

    # -------------------------------------------------
    # Divergence check (cell-centred; the staggered write below is what
    #   the solver actually uses, but this confirms delta_B is clean)
    # -------------------------------------------------
    divB = (np.gradient(BXcc, dx, axis=2)
            + np.gradient(BYcc, dy, axis=1)
            + np.gradient(BZcc, dz, axis=0))
    print("divB RMS =", np.sqrt(np.mean(divB**2)))

    # -------------------------------------------------
    # Elsasser check (uses the existing routine)
    # -------------------------------------------------
    check_elsasser(Hy_grid, BXcc, BYcc, BZcc)

    # -------------------------------------------------
    # Build conserved total energy
    # -------------------------------------------------
    rho = Hy_grid[0]
    vx = Hy_grid[1] / (rho + EPS)
    vy = Hy_grid[2] / (rho + EPS)
    vz = Hy_grid[3] / (rho + EPS)

    kin_e = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    mag_e = 0.5 * (BXcc**2 + BYcc**2 + BZcc**2)
    gas_e = p0 / (gamma - 1.0)
    Hy_grid[4] = kin_e + mag_e + gas_e

    # -------------------------------------------------
    # Sound speed for athinput  (B2 unchanged by velocity masking)
    # -------------------------------------------------
    B2 = BXcc**2 + BYcc**2 + BZcc**2
    mean_rho_on_B2 = box_avg(rho / B2)

    # Adiabatic sound speed for athinput bookkeeping only (solver derives it
    # from primitives at runtime). c_s^2 = gamma * p / rho.
    c_s_squared_mean = box_avg(gamma * p0 / (rho + EPS))
    iso_sound_speed = np.sqrt(c_s_squared_mean)

    #iso_sound_speed = np.sqrt(0.5 * beta / mean_rho_on_B2)

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

    # -------------------------------------------------
    # Write staggered B (via vector potential -> curl) and hydro
    # -------------------------------------------------
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
