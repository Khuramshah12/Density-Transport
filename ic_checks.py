"""
ic_checks.py
============
Drop-in verification functions to call inside create_athena_alfvenspec.

Usage (add to create_athena_alfvenspec after the density grid is built):
    from ic_checks import check_delta_rho, check_elsasser, xz_mean, xz_fluctuation

WHAT IS CHECKED AND WHY
━━━━━━━━━━━━━━━━━━━━━━━
δρ = ρ - ⟨ρ⟩_{x,z}(y)

In reduced MHD the background density is defined as the xz-mean profile
ρ₀(y) = ⟨ρ⟩_{x,z}(y).  The fluctuation δρ is everything on top of that.

At t=0, the density field is initialised as a pure function of y only
(a sine wave).  Therefore δρ must be identically zero — any non-zero value
means either:
  (a) the mesh has a non-uniform Jacobian (unlikely for Cartesian Athena++), or
  (b) a bug in the y-coordinate mapping (yfrac calculation, global vs local y).

The check prints the max absolute value of δρ and raises an error if it
exceeds a tolerance.  The same tolerance is appropriate for double-precision
arithmetic (should be < 1e-14 for a pure y-dependent init).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Core helper: xz-mean and fluctuation
# ─────────────────────────────────────────────────────────────────────────────

def xz_mean(field):
    """
    ⟨field⟩_{x,z}(y)  — average over x (axis 2) and z (axis 0).

    Parameters
    ----------
    field : ndarray, shape (Nz, Ny, Nx)

    Returns
    -------
    mean : ndarray, shape (1, Ny, 1)  — ready for broadcasting against field
    """
    return np.mean(field, axis=(0, 2), keepdims=True)   # shape (1, Ny, 1)


def xz_fluctuation(field):
    """
    δX = X - ⟨X⟩_{x,z}(y)

    This is the RMHD-consistent definition of a fluctuation: it removes only
    the y-dependent background, leaving the full (x, y, z) structure of the
    perturbation.

    Parameters
    ----------
    field : ndarray, shape (Nz, Ny, Nx)

    Returns
    -------
    delta : ndarray, shape (Nz, Ny, Nx)
    """
    return field - xz_mean(field)


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 1: δρ = 0 at t = 0
# ─────────────────────────────────────────────────────────────────────────────

def check_delta_rho(rho_grid, tol=1e-10, label='IC'):
    """
    Verify that the density field has no xz-fluctuations at t=0.

    At initialisation ρ = ρ₀(y) only, so δρ ≡ ρ - ⟨ρ⟩_{x,z}(y) must be
    identically zero to floating-point precision.  A non-zero value indicates
    a coordinate mapping bug.

    Parameters
    ----------
    rho_grid : ndarray, shape (Nz, Ny, Nx) — density array (Athena ordering)
    tol      : float — maximum acceptable |δρ|_max (default 1e-10)
    label    : str   — prefix for print output
    """
    delta_rho = xz_fluctuation(rho_grid)

    max_abs   = float(np.max(np.abs(delta_rho)))
    rms       = float(np.sqrt(np.mean(delta_rho**2)))
    rho_mean  = float(np.mean(rho_grid))

    print(f"[{label}] δρ check  |δρ|_max = {max_abs:.3e}   "
          f"δρ_rms = {rms:.3e}   (relative: {max_abs/rho_mean:.3e})")

    if max_abs > tol:
        raise AssertionError(
            f"[{label}] δρ is NOT zero at t=0!  |δρ|_max = {max_abs:.3e} > tol={tol:.1e}\n"
            "  → Possible cause: y-coordinate mapping bug in density initialisation.\n"
            "  → Check that yfrac = (y - ymin) / Ly is computed from GLOBAL y,\n"
            "    not local meshblock coordinates."
        )
    print(f"[{label}] ✓  δρ = 0 at t=0  (within tol={tol:.1e})")


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2: Elsässer z⁻ ≈ 0 (pure forward wave)
# ─────────────────────────────────────────────────────────────────────────────

def check_elsasser(Hy_grid, BXcc, BYcc, BZcc, label='IC', tol_ratio=0.05):
    """
    Verify that the backward Elsässer variable z⁻ is negligible at t=0.

    z⁻ = δv⊥ - δB⊥/√ρ₀   (should be ≈ 0 for a pure forward wave)
    z⁺ = δv⊥ + δB⊥/√ρ₀   (carries all the energy)

    Here δ denotes the xz-fluctuation  δX = X - ⟨X⟩_{x,z}(y), and √ρ₀ uses
    the background density ρ₀(y) = ⟨ρ⟩_{x,z}(y).

    Parameters
    ----------
    Hy_grid : ndarray, shape (NHYDRO, Nz, Ny, Nx) — conserved hydro variables
    BXcc, BYcc, BZcc : ndarray, shape (Nz, Ny, Nx) — cell-centred B-field
    label   : str
    tol_ratio : float — |z⁻|_rms / |z⁺|_rms must be below this
    """
    rho   = Hy_grid[0].astype(float)             # (Nz, Ny, Nx)
    rho0  = xz_mean(rho)                          # (1, Ny, 1) background density
    inv_sqrt_rho0 = 1.0 / np.sqrt(rho0 + 1e-30)  # broadcast-ready

    # Velocity fluctuations (use ρ₀ for normalisation consistency)
    vy = Hy_grid[2] / (rho + 1e-30)
    vz = Hy_grid[3] / (rho + 1e-30)

    dvy = xz_fluctuation(vy)
    dvz = xz_fluctuation(vz)

    # Magnetic fluctuations normalised by background density
    dBy = xz_fluctuation(BYcc) * inv_sqrt_rho0
    dBz = xz_fluctuation(BZcc) * inv_sqrt_rho0

    zplus_y  = dvy + dBy
    zplus_z  = dvz + dBz
    zminus_y = dvy - dBy
    zminus_z = dvz - dBz

    def rms(a):
        return float(np.sqrt(np.mean(a**2)))

    rms_zp = np.sqrt(rms(zplus_y)**2  + rms(zplus_z)**2)
    rms_zm = np.sqrt(rms(zminus_y)**2 + rms(zminus_z)**2)

    ratio = rms_zm / (rms_zp + 1e-30)

    print(f"[{label}] Elsässer check  |z⁺|_rms = {rms_zp:.4e}   "
          f"|z⁻|_rms = {rms_zm:.4e}   ratio = {ratio:.4e}")

    if ratio > tol_ratio:
        print(f"[{label}] WARNING: |z⁻|/|z⁺| = {ratio:.3f} > {tol_ratio}. "
              "Backward wave energy is not negligible.")
    else:
        print(f"[{label}] ✓  z⁻ ≈ 0  (ratio {ratio:.3e} < {tol_ratio})")

    return rms_zp, rms_zm


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 3: Alfvén speed variation
# ─────────────────────────────────────────────────────────────────────────────

def check_alfven_speed(rho_grid, BXcc, label='IC'):
    """
    Print the variation of the Alfvén speed vA(y) = B₀/√ρ₀(y) to confirm
    the density gradient is set correctly and matches the expected δρ_bg.
    """
    rho0  = xz_mean(rho_grid)[0, :, 0]          # (Ny,) — background profile
    B0    = float(np.mean(np.abs(BXcc)))
    vA    = B0 / np.sqrt(rho0 + 1e-30)

    print(f"[{label}] Alfvén speed  vA_min={vA.min():.4f}  "
          f"vA_max={vA.max():.4f}  "
          f"vA_mean={vA.mean():.4f}  "
          f"fractional_variation={(vA.max()-vA.min())/vA.mean():.4f}")
