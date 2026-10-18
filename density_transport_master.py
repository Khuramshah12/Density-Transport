#!/usr/bin/env python3
"""
================================================================================
 density_transport_master.py  —  v13 "MASTER"
================================================================================
Definitive RMHD density-transport diagnostic suite.  Supersedes and replaces
v8–v12.  Single standalone file; no other project module is required except
``athena_read`` (Athena++ HDF5 reader).

TWO-STAGE ARCHITECTURE (strict reduction/plotting decoupling)
--------------------------------------------------------------------------------
  Stage 1  REDUCE : read every .athdf snapshot once, compute all scalar and
                    profile diagnostics, persist them to
                        OUTPUT_DIR/reduced_data.npz   (arrays)
                        OUTPUT_DIR/reduced_meta.json  (scalars / metadata)
                    Optionally also the shell-binned cross-spectra
                        OUTPUT_DIR/spectra_data.npz + spectra_meta.json
  Stage 2  PLOT   : load the reduced arrays and render any subset of figures.
                    NO snapshot file is touched in this stage, so label sizes,
                    figure dimensions, and styles can be iterated instantly.

USAGE
--------------------------------------------------------------------------------
  python density_transport_master.py reduce [--spectra]
  python density_transport_master.py plot  [--figs F09 F10 ...] [--style poster]
  python density_transport_master.py table            # bridge table only
  python density_transport_master.py all  [--spectra] # reduce + table + plot

PHYSICAL / CODING CONVENTIONS  (single source of truth)
--------------------------------------------------------------------------------
Geometry & coordinates
  * Athena++ arrays are indexed (x3, x2, x1) = (z, y, x):  axis 0 = z,
    axis 1 = y, axis 2 = x.  The background field B0 is along x (parallel
    direction); the imposed density gradient is along y; z completes the
    perpendicular plane.
  * <.>_xz denotes an average over axes (0, 2); profiles are functions of y.
  * All lengths are in units of L_y (= 1); time in Alfvén units t_A = L_x/v_A0.

Fluctuations & Elsasser fields
  * delta(f)   = f - <f>_xz            (mean-profile subtraction, per snapshot)
  * z^pm_perp  = delta(v_perp) +/- delta(B_perp)/sqrt(rho0(y))   [code sign
    convention; RMS amplitudes, energies and sigma_c are invariant under the
    +/- labelling, which depends only on the sign of B0.]
  * z^+_rms    = sqrt(< (z^+_y)^2 + (z^+_z)^2 >)   TWO-COMPONENT VECTOR RMS.
    This is identical to the convention of Squire et al. (2026); there is no
    single-component reduction anywhere in this suite.
  * sigma_c    = (E^+ - E^-)/(E^+ + E^-),  E^pm = (z^pm_rms)^2.

Transport law & closure
  * Flux:        Gamma_rho(y,t) = < delta(rho) delta(v_y) >_xz
  * Predictor:   X(y,t) = - z^+_rms(t) * l_perp,y(t) * d_y rho0(y,t)
  * Law:         Gamma_rho = alpha * X      (no-intercept WLS; see estimator)
  * alpha_th(sigma_c) = 1/[4(1+sigma_c)]  -> 1/8 at sigma_c = 1
  * Lorentzian bridge: alpha_eff(chi_A) = alpha_th * chi_A^2/(1+chi_A^2)
  * chi_A = z^+_rms * l_par / (v_A * l_perp,y).  DEFAULT definition uses the
    box-mode parallel scale l_par = Lx/(2 pi kappa_par) and v_A = 1 (values
    quoted in the manuscript).  A measured-l_par / mean-v_A variant is stored
    alongside and can be enabled globally via CHI_CONFIG (see below).

Correlation lengths
  * l_perp,y = Wiener–Khinchin integral autocorrelation scale of z^+_y along
    y (Fickian mixing scale); l_perp,z analogous along z; l_perp,full is the
    component/direction average used only in the slaved-amplitude ratio.
  * l_par,meas = same ACF construction along x (parallel), averaged over the
    two perpendicular Elsasser components.  Diagnostic only by default.

Numerics
  * Dissipation calibration: X_DISS = 40 for the PLM (piecewise-linear /
    van Leer) reconstruction actually used by the runs.  (Earlier versions
    mislabelled this "PPM"; the value is unchanged.)
  * Plasma beta is now PER GROUP (GroupMeta.beta); v_A^2/c_s^2 = 2/(gamma*beta)
    is computed per run.  All groups currently declare beta = 1.  Low-beta
    runs MUST be added to OUTPUT_SPECS/GROUP_META with their own beta to be
    processed at all.

DELIBERATE NUMERICAL CHANGES vs v12  (verify before adopting numbers)
--------------------------------------------------------------------------------
  (1) Periodic gradient.  d_y rho0 is computed with an exact periodic central
      difference (np.roll) instead of np.gradient with one-sided edges.  The
      stored gradient is used EVERYWHERE (regression included); v12 recomputed
      it inside the regressor with different edge handling.  Affects only the
      2 boundary cells of each 500-point profile.
  (2) Exact Fourier projection.  Harmonic amplitudes use the exact discrete
      projection a_c = 2*mean(f cos), a_s = 2*mean(f sin) on the uniform
      periodic grid, replacing trapezoid quadrature that dropped the wrap
      segment (~0.2% bias at N_y = 500).
  (3) Single alpha estimator.  regress_alpha_from_flux (v9 engine: transient
      cut, SNR mask, |X|^2 weights, bootstrap errors) is the ONLY calibration
      path.  The v10/v11 duo fit_alpha_from_flux / fit_alpha_flux (response-
      magnitude weighting; biased high) is REMOVED, not aliased.
      coarse_erosion_alpha survives strictly as a consistency observable.
  All three shift alpha at or below the bootstrap-error level; regression-test
  bridge_table.csv against the v12 output before updating manuscript numbers.

Author: Khurram Abbas — master refactor v13.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:                                     # SciPy < 1.6
    from scipy.integrate import cumtrapz                # noqa: F401

warnings.filterwarnings('ignore')

sys.path.insert(1, '/khurram/density_transport/'
                   'Initial Conditions/')
import athena_read as aread                             # noqa: E402


# =============================================================================
#  SECTION 1 — CONFIGURATION
# =============================================================================

BASE = ('/khurram/density_transport/500_k15/')

OUTPUT_DIR = BASE + 'unified_diagnostics_results/'

REDUCED_NPZ  = 'reduced_data.npz'
REDUCED_JSON = 'reduced_meta.json'
SPECTRA_NPZ  = 'spectra_data.npz'
SPECTRA_JSON = 'spectra_meta.json'
REPORT_TXT   = 'unified_diagnostics_report.txt'

# Numerical constants -----------------------------------------------------------
X_DISS       = 40          # PLM dissipation calibration (see module docstring)
NY_GRID      = 500
FILE_PATTERN = 'from_array.out.*.athdf'
T_MAX_PLOT   = 50.0
GAMMA_AD     = 5.0 / 3.0

# chi_A definition switch (global, applied suite-wide) ---------------------------
#   use_measured_lpar : replace the box-mode l_par with the measured parallel
#                       ACF scale l_par,meas(t).
#   use_mean_vA       : replace v_A = 1 with <rho0(y)^(-1/2)>_y(t); relevant
#                       for the A_rho = 0.5 square-wave groups (+-22% swing).
# Defaults reproduce the manuscript / v12 numbers exactly.
CHI_CONFIG = dict(use_measured_lpar=False, use_mean_vA=False)


@dataclass(frozen=True)
class GroupMeta:
    """Immutable specification of one simulation family (gid).

    Parameters
    ----------
    regime : {'strong', 'weak', 'reference'}
        Nominal turbulence regime from the run design (chi_A target).
    square : bool
        True for square-wave (broadband tanh-interface) density profiles;
        False for single-harmonic sinusoidal profiles.
    kappa_par, kappa_perp : int
        Parallel / perpendicular mode numbers of the driving Alfvén wave,
        so that k_par = 2*pi*kappa_par/Lx and k_perp = 2*pi*kappa_perp/Ly.
    z0p : float
        Initial z^+ amplitude.
    Lx, Ly, Lz : float
        Box dimensions (code units; Ly = Lz = 1 throughout the suite).
    beta : float
        Plasma beta of the run family.  Enters only through
        v_A^2/c_s^2 = 2/(gamma*beta); dynamics are otherwise beta-agnostic
        by the entropy-mediated slaving argument (manuscript Sec. 2.4).
    ls, lw, label : plotting style + LaTeX legend label.
    """
    regime: str
    square: bool
    kappa_par: int
    kappa_perp: int
    z0p: float
    Lx: float
    Ly: float = 1.0
    Lz: float = 1.0
    beta: float = 1.0
    ls: object = '-'
    lw: float = 2.2
    label: str = ''


GROUP_META: dict[int, GroupMeta] = {
    # --- STRONG --------------------------------------------------------------
    0: GroupMeta('strong', False, 1, 5,  0.20, 5.0,  ls='-',
                 label=r'S1: $L_x{=}5,\;\kappa_\perp{=}5$'),
    1: GroupMeta('strong', False, 1, 10, 0.20, 5.0,  ls=(0, (3, 1, 1, 1)),
                 lw=2.0, label=r'S2: $L_x{=}5,\;\kappa_\perp{=}10$'),
    2: GroupMeta('strong', False, 1, 5,  0.20, 10.0, ls=(0, (3, 5, 1, 5)),
                 label=r'S3: $L_x{=}10,\;\kappa_\perp{=}5$'),
    3: GroupMeta('strong', True,  1, 5,  0.20, 5.0,  ls=(0, (5, 2)),
                 label=r'S4: square, $L_x{=}5,\;\kappa_\perp{=}5$'),
    # --- WEAK ----------------------------------------------------------------
    4: GroupMeta('weak',   False, 1, 5,  0.20, 1.0,  ls='-',
                 label=r'W1: $L_x{=}1,\;\kappa_\perp{=}5$'),
    5: GroupMeta('weak',   False, 2, 5,  0.20, 1.0,  ls='--',
                 label=r'W2: $L_x{=}1,\;\kappa_\parallel{=}2,\;\kappa_\perp{=}5$'),
    6: GroupMeta('weak',   False, 1, 3,  0.075, 5.0, ls=':', lw=2.0,
                 label=r'W3: $L_x{=}5,\;\kappa_\perp{=}3,\;z_0^+{=}0.075$'),
    7: GroupMeta('weak',   True,  1, 5,  0.20, 1.0,  ls=(0, (5, 2)), lw=2.0,
                 label=r'W4: square, $L_x{=}1,\;\kappa_\perp{=}5$'),
    # --- REFERENCE -----------------------------------------------------------
    8: GroupMeta('reference', False, 1, 7, 0.30, 1.0, ls='-.', lw=2.4,
                 label=r'R: $L_x{=}1,\;\kappa_\perp{=}7,\;z_0^+{=}0.3$'),
    # --- LOW-BETA SCAN (add real folders in OUTPUT_SPECS to activate) --------
    # 9:  GroupMeta('strong', False, 1, 5, 0.20, 5.0, beta=0.1,
    #               ls='-', label=r'B1: $\beta{=}0.1$'),
    # 10: GroupMeta('strong', False, 1, 5, 0.20, 5.0, beta=0.01,
    #               ls='--', label=r'B2: $\beta{=}0.01$'),
}

# (A_rho, output folder, gid) — a run is processed IFF it appears here.
OUTPUT_SPECS = [
    # --- STRONG (sinusoidal) --------------------------------------------------
    (0.00,  BASE + 'rho0_Lx5/output/',                       0),
    (0.025, BASE + 'rho0_025_Lx5/output/',                   0),
    (0.05,  BASE + 'rho0_05_Lx5/output/',                    0),
    (0.025, BASE + 'A02k110/rho0_025/output/',               1),
    (0.05,  BASE + 'A02k110/rho0_05/output/',                1),
    (0.05,  BASE + 'rho0_05_Lx10/output/',                   2),
    (0.10,  BASE + 'rho0_1_Lx10/output/',                    2),
    # --- STRONG square-wave (S4) ----------------------------------------------
    (0.5,   BASE + 'Sharp_grad/rho0_5_kp15B01A02/output/',   3),
    # --- WEAK (sinusoidal) ------------------------------------------------------
    (0.00,  BASE + 'rho0/output/',                           4),
    (0.025, BASE + 'rho0_025/output/',                       4),
    (0.05,  BASE + 'rho0_05/output/',                        4),
    (0.05,  BASE + 'rho0_05A02kp2/output/',                  5),
    (0.10,  BASE + 'rho0_1A02kp2/output/',                   5),
    (0.025, BASE + 'test_Lx5kp3z0075/output/',               6),
    # --- WEAK square-wave (W4) ---------------------------------------------------
    (0.5,   BASE + 'Sharp_grad/cubic_rho0_5kp15B01A02/output/', 7),
    # --- REFERENCE ----------------------------------------------------------------
    (0.25,  BASE + 'cubic_test_z03rho025kp7/output/',        8),
]

STRONG_GIDS    = [g for g, m in GROUP_META.items() if m.regime == 'strong']
WEAK_GIDS      = [g for g, m in GROUP_META.items() if m.regime == 'weak']
REFERENCE_GIDS = [g for g, m in GROUP_META.items() if m.regime == 'reference']
ACTIVE_GIDS_2x4 = (STRONG_GIDS + WEAK_GIDS)[:8]
ACTIVE_GIDS_ALL = STRONG_GIDS + WEAK_GIDS + REFERENCE_GIDS


# --- regime / profile helpers ----------------------------------------------------

def regime_of(gid: int) -> str:  return GROUP_META[gid].regime
def is_strong(gid) -> bool:      return regime_of(gid) == 'strong'
def is_weak(gid) -> bool:        return regime_of(gid) == 'weak'
def is_reference(gid) -> bool:   return regime_of(gid) == 'reference'
def is_square(gid) -> bool:      return GROUP_META[gid].square


def short_id(gid: int) -> str:
    """Compact family identifier, e.g. 'S1', 'W4', 'R'."""
    return GROUP_META[gid].label.split(':')[0].strip()


def panel_title(gid: int, A_rho: float | None = None) -> str:
    sid = short_id(gid)
    if A_rho is None:
        return rf'$\mathtt{{{sid}}}$'
    return rf'$\mathtt{{{sid}\_A{A_rho:g}}}$'


# --- geometry / timescales ---------------------------------------------------------

def k_par_phys_of(gid):
    gm = GROUP_META[gid]
    return gm.kappa_par * 2.0 * np.pi / gm.Lx


def k_perp_phys_of(gid):
    gm = GROUP_META[gid]
    return gm.kappa_perp * 2.0 * np.pi / gm.Ly


def l_par_phys_of(gid):
    """Box-mode parallel scale l_par = 1/k_par = Lx/(2 pi kappa_par)."""
    return 1.0 / k_par_phys_of(gid)


def k_diss_of(gid, X=X_DISS):
    """PLM dissipation wavenumber k_diss = 2 pi N_y / (X Ly), X = 40."""
    return 2.0 * np.pi * NY_GRID / (X * GROUP_META[gid].Ly)


def t_diss_inst(gid, dvAdy_max, X=X_DISS):
    """Instantaneous phase-mixing time to reach k_diss from k_par*|d_y v_A|."""
    if dvAdy_max < 1e-12:
        return np.inf
    return k_diss_of(gid, X) / (k_par_phys_of(gid) * dvAdy_max)


def va2_over_cs2_of(gid):
    """v_A^2/c_s^2 = 2/(gamma beta), PER GROUP (beta from GroupMeta)."""
    return 2.0 / (GAMMA_AD * GROUP_META[gid].beta)


# =============================================================================
#  SECTION 2 — NUMERICAL PRIMITIVES
# =============================================================================

def _trapz(y, x):
    """np.trapezoid with fallback for old NumPy."""
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def periodic_gradient(f, y, axis=-1):
    """Exact periodic central difference df/dy on a uniform grid.

    Uses (f[i+1] - f[i-1]) / (2 dy) with wrap-around indices, which is the
    correct stencil for periodic data; np.gradient's one-sided edge formulas
    would introduce O(dy) boundary artefacts inconsistent between call sites
    (v12 defect (1) in the module docstring).

    Parameters
    ----------
    f : ndarray            Field sampled on the uniform periodic grid.
    y : ndarray (1D)       Coordinate values (only the spacing is used).
    axis : int             Axis along which to differentiate.
    """
    dy = float(y[1] - y[0])
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dy)


def fourier_mode_amplitude(profile, y_arr, Ly, n):
    """Exact discrete amplitude of harmonic n of a periodic 1D profile.

    For a uniform periodic sampling y_j the discrete projection
        a_c = (2/N) sum_j f_j cos(2 pi n y_j / Ly)   (and a_s analogously)
    equals the exact continuum Fourier coefficient of the band-limited
    interpolant (rectangle rule = exact on periodic data).  Trapezoid
    quadrature over [y_0, y_{N-1}] used in v <= 12 drops the wrap segment
    and biases amplitudes low by O(1/N) (module docstring, change (2)).

    Returns
    -------
    float : sqrt(a_c^2 + a_s^2)  (phase-independent harmonic amplitude)
    """
    theta = 2.0 * np.pi * n * np.asarray(y_arr) / Ly
    a_c = 2.0 * float(np.mean(profile * np.cos(theta)))
    a_s = 2.0 * float(np.mean(profile * np.sin(theta)))
    return float(np.hypot(a_c, a_s))


def _acf_len_1d(field, axis, L):
    """Integral autocorrelation length along one axis (Wiener–Khinchin).

    Method
    ------
    1. Remove the mean along `axis` (fluctuation field).
    2. Power spectrum P(k) = <|FFT|^2>/N^2, averaged over the other axes.
       By the Wiener–Khinchin theorem the inverse FFT of P(k) is the
       (periodic) autocorrelation function R(lag).
    3. Normalise R by R(0) and integrate from lag 0 to the first zero
       crossing (or N/2 if R stays positive):  l = int_0^{lag_0} R d(lag).

    The zero-crossing cutoff avoids the negative-lobe cancellation that
    makes the full-period integral of a periodic ACF vanish identically.

    Returns np.nan if the field has no variance along `axis`.
    """
    N    = field.shape[axis]
    lags = np.arange(N) * (L / N)
    f    = field - np.mean(field, axis=axis, keepdims=True)
    Pk   = np.mean(np.abs(np.fft.fft(f, axis=axis)) ** 2,
                   axis=tuple(i for i in range(field.ndim) if i != axis)) / N**2
    ac   = np.real(np.fft.ifft(Pk))
    if ac[0] <= 0:
        return np.nan
    R   = ac / ac[0]
    neg = np.where(R[1:N // 2] <= 0)[0]
    zc  = int(neg[0]) + 1 if len(neg) else N // 2
    return max(_trapz(R[:zc + 1], lags[:zc + 1]), 0.0)


# =============================================================================
#  SECTION 3 — REDUCTION ENGINE  (Stage 1; the ONLY code that reads .athdf)
# =============================================================================

def process_snapshot(fn, gid):
    """Reduce one .athdf snapshot to scalar and 1D-profile diagnostics.

    Returns a dict with (see module docstring for conventions):

    Scalars
      t, rho_mean, zplus_rms, zminus_rms, sigma_c, dvpar_rms, dBpar_rms,
      drho_over_rho0_rms, rho_vy_corr, dvAdy_max, Frho_rms_y, Frho_max_y,
      profile_var_norm, rho_power_k{2,3,4}, A_rho_k1, A_rho_k2,
      l_perp_y, l_perp_z, l_perp_{y,z}_full, l_perp_full,
      l_par_meas  (parallel ACF scale, avg of z^+_y and z^+_z along x),
      vA_mean     (<rho0(y)^{-1/2}>_y, for the CHI_CONFIG mean-v_A option).

    Profiles (length N_y)
      y_arr, rho0, rho_norm, drho0_dy (exact periodic gradient), Gamma_rho.
    """
    d  = aread.athdf(fn)
    t  = float(d['Time'])
    gm = GROUP_META[gid]

    rho = np.asarray(d['rho'],  dtype=np.float64)
    vy  = np.asarray(d['vel2'], dtype=np.float64)
    vz  = np.asarray(d['vel3'], dtype=np.float64)
    By  = np.asarray(d['Bcc2'], dtype=np.float64)
    Bz  = np.asarray(d['Bcc3'], dtype=np.float64)
    vx  = np.asarray(d['vel1'], dtype=np.float64)
    Bx  = np.asarray(d['Bcc1'], dtype=np.float64)

    xz = (0, 2)                                   # average over (z, x)
    try:
        ymin = float(d['RootGridX2'][0]); ymax = float(d['RootGridX2'][1])
    except Exception:
        ymin, ymax = 0.0, gm.Ly
    Ly = ymax - ymin
    y_arr = ymin + (np.arange(rho.shape[1]) + 0.5) * (Ly / rho.shape[1])
    try:
        Lz = float(d['RootGridX3'][1]) - float(d['RootGridX3'][0])
    except Exception:
        Lz = gm.Lz
    Lx = gm.Lx

    # --- mean profile and its derivatives -----------------------------------
    rho0_3d  = np.mean(rho, axis=xz, keepdims=True)
    rho0_1d  = rho0_3d[0, :, 0]
    rho_mean = float(np.mean(rho0_1d))
    rho_norm = rho0_1d / (rho_mean + 1e-30) - 1.0

    vA_y      = 1.0 / np.sqrt(rho0_1d + 1e-30)
    vA_mean   = float(np.mean(vA_y))
    dvAdy_max = float(np.max(np.abs(periodic_gradient(vA_y, y_arr))))

    grad_log_rho0 = periodic_gradient(np.log(rho0_1d + 1e-30), y_arr)
    grad_rho0     = periodic_gradient(rho0_1d, y_arr)
    Frho_rms_y = float(np.sqrt(np.mean(grad_log_rho0 ** 2)))
    Frho_max_y = float(np.max(np.abs(grad_log_rho0)))

    profile_var_norm = float(np.var(rho_norm))
    A_rho_k1 = fourier_mode_amplitude(rho_norm, y_arr, Ly, 1)
    A_rho_k2 = fourier_mode_amplitude(rho_norm, y_arr, Ly, 2)
    rho_power_k2 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 2) ** 2)
    rho_power_k3 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 3) ** 2)
    rho_power_k4 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 4) ** 2)

    # --- fluctuations and Elsasser fields ------------------------------------
    inv_sqr   = 1.0 / np.sqrt(rho0_3d + 1e-30)
    delta_vy  = vy  - np.mean(vy,  axis=xz, keepdims=True)
    delta_vz  = vz  - np.mean(vz,  axis=xz, keepdims=True)
    delta_vx  = vx  - np.mean(vx,  axis=xz, keepdims=True)
    delta_By  = By  - np.mean(By,  axis=xz, keepdims=True)
    delta_Bz  = Bz  - np.mean(Bz,  axis=xz, keepdims=True)
    delta_Bx  = Bx  - np.mean(Bx,  axis=xz, keepdims=True)
    delta_rho = rho - rho0_3d

    zp_y = delta_vy + delta_By * inv_sqr
    zp_z = delta_vz + delta_Bz * inv_sqr
    zm_y = delta_vy - delta_By * inv_sqr
    zm_z = delta_vz - delta_Bz * inv_sqr

    zplus_rms  = float(np.sqrt(np.mean(zp_y**2 + zp_z**2)))
    zminus_rms = float(np.sqrt(np.mean(zm_y**2 + zm_z**2)))
    E_p, E_m = zplus_rms**2, zminus_rms**2
    sigma_c = (E_p - E_m) / (E_p + E_m) if (E_p + E_m) > 1e-30 else 0.0

    dvpar_rms = float(np.sqrt(np.mean(delta_vx**2)))
    dBpar_rms = float(np.sqrt(np.mean(delta_Bx**2)))

    drho_over_rho0     = delta_rho / (rho0_3d + 1e-30)
    drho_over_rho0_rms = float(np.sqrt(np.mean(drho_over_rho0**2)))

    # global real-space correlation r(delta_rho, delta_vy)
    num = float(np.mean(delta_rho * delta_vy))
    den = float(np.sqrt(np.mean(delta_rho**2) * np.mean(delta_vy**2)) + 1e-60)
    rho_vy_corr = num / den

    # cross-field turbulent flux profile
    Gamma_rho_1d = np.mean(delta_rho * delta_vy, axis=xz)

    # --- correlation lengths --------------------------------------------------
    acf_y_zpy = _acf_len_1d(zp_y, axis=1, L=Ly)
    acf_y_zpz = _acf_len_1d(zp_z, axis=1, L=Ly)
    acf_z_zpy = _acf_len_1d(zp_y, axis=0, L=Lz)
    acf_z_zpz = _acf_len_1d(zp_z, axis=0, L=Lz)
    acf_x_zpy = _acf_len_1d(zp_y, axis=2, L=Lx)      # parallel (measured)
    acf_x_zpz = _acf_len_1d(zp_z, axis=2, L=Lx)

    def _avg(a, b):
        if np.isfinite(a) and np.isfinite(b):
            return 0.5 * (a + b)
        return a if np.isfinite(a) else b

    l_perp_y_full = _avg(acf_y_zpy, acf_y_zpz)
    l_perp_z_full = _avg(acf_z_zpy, acf_z_zpz)
    l_perp_full   = _avg(l_perp_y_full, l_perp_z_full)
    l_par_meas    = _avg(acf_x_zpy, acf_x_zpz)

    del (rho, vy, vz, By, Bz, vx, Bx, delta_rho, delta_vy, delta_vz, delta_vx,
         delta_By, delta_Bz, delta_Bx, zp_y, zp_z, zm_y, zm_z, inv_sqr,
         drho_over_rho0)
    gc.collect()

    return dict(
        t=t, Ly=Ly, Lz=Lz, y_arr=y_arr,
        rho0=rho0_1d, rho_mean=rho_mean, rho_norm=rho_norm,
        drho0_dy=grad_rho0,
        profile_var_norm=profile_var_norm,
        rho_power_k2=rho_power_k2, rho_power_k3=rho_power_k3,
        rho_power_k4=rho_power_k4,
        dvAdy_max=dvAdy_max, vA_mean=vA_mean,
        Frho_rms_y=Frho_rms_y, Frho_max_y=Frho_max_y,
        zplus_rms=zplus_rms, zminus_rms=zminus_rms, sigma_c=sigma_c,
        dvpar_rms=dvpar_rms, dBpar_rms=dBpar_rms,
        drho_over_rho0_rms=drho_over_rho0_rms, rho_vy_corr=rho_vy_corr,
        Gamma_rho=Gamma_rho_1d,
        A_rho_k1=A_rho_k1, A_rho_k2=A_rho_k2,
        l_perp_y=acf_y_zpy, l_perp_z=acf_z_zpy,
        l_perp_y_full=l_perp_y_full, l_perp_z_full=l_perp_z_full,
        l_perp_full=l_perp_full, l_par_meas=l_par_meas,
    )


# 1D time-series keys persisted per run (order-independent) ----------------------
_SERIES_KEYS = [
    't', 'A_rho', 'A_rho_norm', 'A_rho_k2', 'A_rho_k2_norm',
    'zplus_rms', 'zminus_rms', 'sigma_c',
    'l_perp_y', 'l_perp_z', 'l_perp_y_full', 'l_perp_z_full', 'l_perp_full',
    'l_par_meas', 'vA_mean', 'dvAdy_max',
    'Frho_rms_y', 'Frho_max_y', 'dvpar_rms', 'dBpar_rms',
    'drho_over_rho0_rms', 'rho_vy_corr', 'profile_var_norm',
    'rho_power_k2', 'rho_power_k3', 'rho_power_k4',
]
_PROFILE_KEYS = ['drho0_dy', 'rho0', 'Gamma_rho']       # 2D (Nt, Ny)
_AXIS_KEYS    = ['y_arr']                               # 1D (Ny)


def _meta_for(gid, A_rho, Ly_measured, A0):
    """Assemble the per-run metadata dict (GROUP_META = single source)."""
    gm = GROUP_META[gid]
    return dict(
        A_rho=float(A_rho), gid=int(gid), regime=gm.regime,
        square=bool(gm.square), beta=float(gm.beta),
        kappa_par=gm.kappa_par, kappa_perp=gm.kappa_perp,
        k_par_phys=k_par_phys_of(gid), k_perp_phys=k_perp_phys_of(gid),
        l_par_phys=l_par_phys_of(gid), k_diss=k_diss_of(gid),
        va2_over_cs2=va2_over_cs2_of(gid),
        z0p=gm.z0p, Lx=gm.Lx, Ly=float(Ly_measured),
        group_label=gm.label, A0=float(A0),
    )


def load_all_runs():
    """Stage-1 reduction: read every run in OUTPUT_SPECS from disk.

    Returns
    -------
    all_data : dict tag -> run dict (meta + arrays; structure documented in
               _SERIES_KEYS / _PROFILE_KEYS / _AXIS_KEYS above).
    cat      : dict tag -> dict(snapshots=list, gid, A_rho, folder); needed
               only by the optional spectra reduction in the same stage.
    load_log : list of human-readable warnings (skipped runs / snapshots).
    """
    all_data, cat, load_log = {}, {}, []

    for A_rho, folder, gid in OUTPUT_SPECS:
        if gid not in GROUP_META:
            load_log.append(f'[SKIP] gid={gid} not in GROUP_META ({folder})')
            continue
        tag = f'g{gid}_Arho{A_rho:.3f}'
        if not os.path.isdir(folder):
            load_log.append(f'[SKIP] {tag}: folder not found ({folder})')
            continue
        snaps = sorted(glob.glob(os.path.join(folder, FILE_PATTERN)))
        if not snaps:
            load_log.append(f'[SKIP] {tag}: no .athdf files in {folder}')
            continue

        print(f'\nLoading {tag}  [{regime_of(gid).upper():>9s}'
              f'{" SQ" if is_square(gid) else "":>3s}]  ({len(snaps)} snapshots)')
        records = []
        for i, fn in enumerate(snaps):
            try:
                records.append(process_snapshot(fn, gid))
                if (i + 1) % 20 == 0 or i == len(snaps) - 1:
                    r = records[-1]
                    print(f'  {i+1}/{len(snaps)}  t={r["t"]:.2f}  '
                          f'z+={r["zplus_rms"]:.4f}  l_y={r["l_perp_y"]:.4f}')
            except Exception as exc:
                msg = f'[WARN] {tag}: skipped {os.path.basename(fn)}: {exc}'
                print('  ' + msg); load_log.append(msg)

        if not records:
            load_log.append(f'[SKIP] {tag}: zero readable snapshots')
            continue
        records.sort(key=lambda r: r['t'])

        def _arr(key):
            return np.array([r[key] for r in records])

        A_k1 = _arr('A_rho_k1')
        A0   = A_k1[0] if A_k1[0] > 1e-12 else 1.0

        run = dict(meta=_meta_for(gid, A_rho, records[0]['Ly'], A0))
        run['t']            = _arr('t')
        run['A_rho']        = A_k1
        run['A_rho_norm']   = A_k1 / A0
        run['A_rho_k2']     = _arr('A_rho_k2')
        run['A_rho_k2_norm'] = run['A_rho_k2'] / A0     # NOTE: normalised to k1(0)
        for key in _SERIES_KEYS:
            if key in run:                              # already assembled above
                continue
            run[key] = _arr(key)
        for key in _PROFILE_KEYS:
            run[key] = np.vstack([r[key] for r in records])
        run['y_arr'] = records[0]['y_arr']
        run['Ly']    = records[0]['Ly']

        all_data[tag] = run
        cat[tag] = dict(snapshots=snaps, gid=gid, A_rho=A_rho, folder=folder)
        del records; gc.collect()

    return all_data, cat, load_log


# --- persistence (the reduce/plot isolation layer) -------------------------------

def _safe_key(tag):
    return tag.replace('.', 'p').replace('-', 'n').replace(' ', '_')


def save_reduced(all_data, out_dir=OUTPUT_DIR):
    """Persist reduced arrays (npz) and metadata (json)."""
    os.makedirs(out_dir, exist_ok=True)
    flat, meta = {}, {}
    for tag, run in all_data.items():
        sk = _safe_key(tag)
        meta[tag] = run['meta']
        for key in _SERIES_KEYS + _PROFILE_KEYS + _AXIS_KEYS:
            flat[f'{sk}__{key}'] = np.asarray(run[key])
    np.savez_compressed(os.path.join(out_dir, REDUCED_NPZ), **flat)
    with open(os.path.join(out_dir, REDUCED_JSON), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  -> reduced data : {os.path.join(out_dir, REDUCED_NPZ)}')
    print(f'  -> reduced meta : {os.path.join(out_dir, REDUCED_JSON)}')


def load_reduced(out_dir=OUTPUT_DIR):
    """Load the reduced dataset written by save_reduced (Stage 2 entry)."""
    npz_path  = os.path.join(out_dir, REDUCED_NPZ)
    json_path = os.path.join(out_dir, REDUCED_JSON)
    if not (os.path.isfile(npz_path) and os.path.isfile(json_path)):
        raise FileNotFoundError(
            f'Reduced data not found in {out_dir}. Run the "reduce" stage first.')
    with open(json_path) as f:
        meta = json.load(f)
    z = np.load(npz_path)
    all_data = {}
    for tag, m in meta.items():
        sk = _safe_key(tag)
        run = dict(meta=m, Ly=float(m['Ly']))
        for key in _SERIES_KEYS + _PROFILE_KEYS + _AXIS_KEYS:
            run[key] = z[f'{sk}__{key}']
        all_data[tag] = run
    return all_data


# =============================================================================
#  SECTION 4 — CLOSURE PHYSICS  (pure functions of reduced arrays)
# =============================================================================

def chi_A(run, config=None):
    """Turbulence-strength parameter chi_A(t) = z^+_rms l_par / (v_A l_perp,y).

    The definition is controlled by CHI_CONFIG (or an override dict):
      * l_par : box-mode Lx/(2 pi kappa_par) [default, manuscript values] or
                the measured parallel ACF scale l_par_meas(t).
      * v_A   : 1 (code units) [default] or <rho0^{-1/2}>_y(t), which matters
                at the tens-of-percent level for the A_rho = 0.5 groups.
    """
    cfg = CHI_CONFIG if config is None else config
    zp  = run['zplus_rms']
    lpy = np.where(run['l_perp_y'] > 0, run['l_perp_y'], np.nan)
    if cfg.get('use_measured_lpar', False):
        l_par = np.where(run['l_par_meas'] > 0, run['l_par_meas'], np.nan)
    else:
        l_par = run['meta']['l_par_phys']
    vA = run['vA_mean'] if cfg.get('use_mean_vA', False) else 1.0
    return zp * l_par / (vA * lpy)


def chi_A_mean(run, config=None):
    return float(np.nanmean(chi_A(run, config)))


def alpha_th_imbalance(sigma_c):
    """Strong-turbulence closure prefactor alpha_th = 1/[4(1+sigma_c)]."""
    return 1.0 / (4.0 * (1.0 + sigma_c))


def lorentzian_alpha(chi, alpha_th=0.125):
    """Parameter-free Lorentzian bridge alpha_eff = alpha_th chi^2/(1+chi^2).

    Used both for the theory curve in F10 and (via build_prediction) for the
    erosion predictors, so the plotted curve and the predictor share one
    implementation.
    """
    chi = np.asarray(chi, dtype=float)
    return alpha_th * chi**2 / (1.0 + chi**2)


def build_prediction(run, alpha_const, mode_n=1, suppression='none'):
    """Fickian erosion prediction for harmonic k = mode_n of the mean profile.

    Integrates  A_n(t)/A_n(0) = exp[-(n K_y)^2 \\int_0^t eta_turb dt'],  with
      eta_turb(t) = alpha_const * S(chi_A) * z^+_rms(t) * l_perp,y(t),
    where the suppression factor S is
      'none'       : 1                       (constant-alpha strong law)
      'lorentzian' : chi^2/(1+chi^2)         (full bridge, Eq. master)
      'weak'       : chi^2                   (chi_A << 1 asymptote)

    Returns (I_cum, A_pred, eta_arr).
    """
    t   = run['t']
    zp  = run['zplus_rms']
    lpy = run['l_perp_y']
    Ky  = mode_n * 2.0 * np.pi / run['Ly']
    lp  = np.where(np.isfinite(lpy) & (lpy > 0), lpy, 0.0)
    if suppression == 'none':
        S = 1.0
    else:
        chi = chi_A(run)
        cs  = np.where(np.isfinite(chi), chi, 0.0)
        S   = cs**2 / (1.0 + cs**2) if suppression == 'lorentzian' else cs**2
    eta = alpha_const * S * zp * lp
    I = np.zeros_like(t); I[1:] = cumtrapz(eta, t)
    return I, np.exp(-Ky**2 * I), eta


def Kperp_phase_mixing(run):
    """Phase-mixing wavenumber K_perp(t) = k_perp,0 + int k_par |d_y v_A| dt."""
    m = run['meta']; t = run['t']
    rate = m['k_par_phys'] * run['dvAdy_max']
    integ = np.zeros_like(t); integ[1:] = cumtrapz(rate, t)
    return m['k_perp_phys'] + integ


def integrate_abs_flux(run):
    """int |Gamma_rho(y,t)| dy as a function of time."""
    y = run['y_arr']
    return np.array([_trapz(np.abs(g), y) for g in run['Gamma_rho']])


# =============================================================================
#  SECTION 5 — PRIMARY ALPHA ESTIMATOR  (single calibration path)
# =============================================================================
#  Rationale for this estimator over the removed v10/v11 variants:
#  weighting must depend on the PREDICTOR magnitude |X| only.  Weighting by
#  the response |Y| = |alpha X + noise| correlates weights with the noise
#  realisation and biases the no-intercept slope high; that bias, not physics,
#  produced part of the flux-vs-erosion alpha offset seen in earlier builds.
# =============================================================================

def _regressor_arrays(run):
    """Predictor X = -z^+ l_perp,y d_y rho0 and response Y = Gamma_rho.

    Both shaped (Nt, Ny).  Uses the STORED periodic gradient from the
    reduction stage — the regression and every figure therefore see one and
    the same d_y rho0 (v12 recomputed it here with different edge handling).
    """
    zp  = run['zplus_rms'][:, None]
    lpy = run['l_perp_y'][:, None]
    X = -zp * lpy * run['drho0_dy']
    Y = run['Gamma_rho']
    valid_t = np.isfinite(run['l_perp_y']) & (run['l_perp_y'] > 0)
    valid = np.broadcast_to(valid_t[:, None], X.shape).copy()
    return X, Y, valid


def regress_alpha_from_flux(run, t_skip_frac=0.05, snr_frac=0.05,
                            weight_power=2.0, n_bootstrap=200, rng_seed=42):
    """Weighted no-intercept WLS slope of Gamma_rho = alpha * X on the (y,t)
    cloud.  THE single source of every quoted alpha in the suite.

    Robustness features (reviewer checklist)
    ----------------------------------------
    * Transient cut: the first ceil(t_skip_frac * Nt) snapshots are excluded
      (initial-condition relaxation; one eddy turnover at default cadence).
    * SNR mask: points with |X| < snr_frac * max|X| are excluded, so the fit
      is controlled by the coherent-gradient regions, not near-node noise.
    * Weights w = |X|^weight_power (predictor-magnitude only; see rationale).
    * Uncertainty: (i) analytic WLS standard error, (ii) case-resampling
      bootstrap (n_bootstrap draws, fixed seed for reproducibility).
    * Weighted R^2 quantifies how much of the flux variance the one-parameter
      law captures; low R^2 in weak runs is expected and reported, not hidden.

    Returns
    -------
    dict with alpha_fit, alpha_err (bootstrap), alpha_err_wls, residual_rms,
    r_squared, n_used, n_total, t_window, status, and the X, Y, mask arrays
    for plotting.
    """
    X, Y, valid = _regressor_arrays(run)
    t = run['t']; Nt = X.shape[0]
    t_skip = int(np.ceil(t_skip_frac * Nt))
    valid[:t_skip, :] = False

    X_abs_max = np.max(np.abs(X[valid])) if valid.any() else 0.0
    keep = valid & (np.abs(X) >= snr_frac * X_abs_max)

    n_total = int(valid.sum()); n_used = int(keep.sum())
    t_window = (float(t[min(t_skip, Nt - 1)]), float(t[-1]))

    if n_used < 10:
        return dict(alpha_fit=np.nan, alpha_err=np.nan, alpha_err_wls=np.nan,
                    residual_rms=np.nan, r_squared=np.nan, n_used=n_used,
                    n_total=n_total, t_window=t_window,
                    status='insufficient_points', X=X, Y=Y, mask=keep)

    Xk, Yk = X[keep], Y[keep]
    Wk = np.abs(Xk) ** weight_power
    Sxx = np.sum(Wk * Xk * Xk); Sxy = np.sum(Wk * Xk * Yk)
    alpha_fit = Sxy / Sxx
    resid = Yk - alpha_fit * Xk
    residual_rms = float(np.sqrt(np.mean(resid**2)))
    sigma2_w = np.sum(Wk * resid**2) / max(np.sum(Wk) - 1.0, 1.0)
    alpha_err_wls = float(np.sqrt(sigma2_w / Sxx))
    Yw = np.sum(Wk * Yk) / np.sum(Wk)
    SS_tot = np.sum(Wk * (Yk - Yw)**2); SS_res = np.sum(Wk * resid**2)
    r2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else np.nan

    rng = np.random.default_rng(rng_seed)
    n = len(Xk); boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = (np.sum(Wk[idx] * Xk[idx] * Yk[idx])
                   / np.sum(Wk[idx] * Xk[idx] * Xk[idx]))
    alpha_err = float(np.std(boot, ddof=1))

    return dict(alpha_fit=float(alpha_fit), alpha_err=alpha_err,
                alpha_err_wls=alpha_err_wls, residual_rms=residual_rms,
                r_squared=float(r2), n_used=n_used, n_total=n_total,
                t_window=t_window, status='ok', X=X, Y=Y, mask=keep)


def regress_alpha_per_time(run, snr_frac=0.05, weight_power=2.0):
    """Per-snapshot alpha_inst(t) from the spatial (y-only) regression.

    Diagnostic companion to regress_alpha_from_flux: shows when the transport
    law switches on and justifies the transient cut visually (fig F05).
    """
    X, Y, valid = _regressor_arrays(run)
    Nt = X.shape[0]; alpha_t = np.full(Nt, np.nan)
    for i in range(Nt):
        if not valid[i].any():
            continue
        Xi, Yi = X[i], Y[i]
        Xmax = np.max(np.abs(Xi))
        if Xmax <= 0:
            continue
        keep = np.abs(Xi) >= snr_frac * Xmax
        if keep.sum() < 4:
            continue
        Wi = np.abs(Xi[keep]) ** weight_power
        Sxx = np.sum(Wi * Xi[keep]**2); Sxy = np.sum(Wi * Xi[keep] * Yi[keep])
        if Sxx > 0:
            alpha_t[i] = Sxy / Sxx
    return alpha_t


def alpha_flux(run):
    """(alpha, rms) convenience wrapper around the primary estimator."""
    fit = regress_alpha_from_flux(run)
    return fit['alpha_fit'], fit['residual_rms']


def coarse_erosion_alpha(run):
    """Apparent alpha from the k=1 amplitude decay rate.

    CONSISTENCY OBSERVABLE ONLY.  It maps the instantaneous logarithmic decay
    of A_rho,k=1 through the Fickian relation; it is noisy (time derivative of
    a noisy amplitude) and is never used to calibrate the transport law.
    """
    t = run['t']; A = run['A_rho']
    zp = run['zplus_rms']; lpy = run['l_perp_y']
    Ky = 2.0 * np.pi / run['Ly']
    logA = np.log(np.maximum(np.abs(A), 1e-16))
    dlogA = np.gradient(logA, t)
    denom = Ky**2 * zp * np.where(lpy > 0, lpy, np.nan)
    return -dlogA / denom


# =============================================================================
#  SECTION 6 — MORPHOLOGY
# =============================================================================

def compute_morphology(run, kmodes=(1, 2, 3, 4)):
    """Shape diagnostics of the mean profile, independent of the flux fit.

    Returns dict with: t, var(t), A_modes[k](t) (exact discrete projections),
    delta_grad(t) = range/max|gradient| (interface thickness proxy), and the
    squared-deviation spread(t) (second moment of (rho0 - <rho0>)^2 about its
    centroid).
    """
    t = run['t']; y = run['y_arr']; rho_yt = run['rho0']; Ly = run['Ly']
    rho_m = np.mean(rho_yt, axis=1)
    var_t = np.var(rho_yt, axis=1)
    norm = rho_yt / rho_m[:, None] - 1.0
    A_modes = {}
    for k in kmodes:
        theta = 2.0 * np.pi * k * y / Ly
        As = 2.0 * np.mean(norm * np.sin(theta)[None, :], axis=1)
        Ac = 2.0 * np.mean(norm * np.cos(theta)[None, :], axis=1)
        A_modes[k] = np.hypot(As, Ac)
    drho_dy = periodic_gradient(rho_yt, y, axis=1)
    rng_t  = rho_yt.max(axis=1) - rho_yt.min(axis=1)
    grad_t = np.max(np.abs(drho_dy), axis=1)
    delta_g = np.where(grad_t > 1e-14, rng_t / grad_t, np.nan)
    dev = rho_yt - rho_m[:, None]; w = dev**2
    W = np.sum(w, axis=1, keepdims=True); y_b = y[None, :]
    y_c = np.where(W > 0, np.sum(w * y_b, axis=1, keepdims=True) / W, 0.5 * Ly)
    spread = np.where(W[:, 0] > 0,
                      np.sqrt(np.sum(w * (y_b - y_c)**2, axis=1) / W[:, 0]),
                      np.nan)
    return dict(t=t, var=var_t, A_modes=A_modes, delta_grad=delta_g,
                spread=spread)


# =============================================================================
#  SECTION 7 — SPECTRA REDUCTION  (optional Stage-1 add-on; needed by F17/F18)
# =============================================================================

def _spec_accumulators(rho_slice, vy_slice, Ly, Lz, bin_edges):
    """Shell-summed auto/cross spectra of (drho/rho0, dv_y) on one x-plane.

    2D rFFT over the (z, y) plane; modes are binned into logarithmic
    k_perp = sqrt(k_y^2 + k_z^2) shells.  Returns shell sums of
    P_rr = |F_rho|^2, P_vv = |F_v|^2, Re/Im of the cross spectrum
    P_rv = F_rho conj(F_v), and mode counts.
    """
    NZ, NY = rho_slice.shape
    Frho = np.fft.rfft2(rho_slice); Fvy = np.fft.rfft2(vy_slice)
    ky = np.fft.rfftfreq(NY, d=Ly / NY) * 2.0 * np.pi
    kz = np.fft.fftfreq(NZ,  d=Lz / NZ) * 2.0 * np.pi
    KY, KZ = np.meshgrid(ky, kz)
    KP = np.sqrt(KY**2 + KZ**2).ravel()
    P_rr = (np.abs(Frho)**2).ravel()
    P_vv = (np.abs(Fvy)**2).ravel()
    P_rv = (Frho * np.conj(Fvy)).ravel()
    s_rr, _ = np.histogram(KP, bins=bin_edges, weights=P_rr)
    s_vv, _ = np.histogram(KP, bins=bin_edges, weights=P_vv)
    s_re, _ = np.histogram(KP, bins=bin_edges, weights=P_rv.real)
    s_im, _ = np.histogram(KP, bins=bin_edges, weights=P_rv.imag)
    cnt,  _ = np.histogram(KP, bins=bin_edges)
    return s_rr, s_vv, s_re, s_im, cnt


def _weighted_shell_phase_and_r(s_rr, s_vv, s_re, s_im,
                                coh_min=0.12, power_frac=1e-3):
    """Power-weighted shell phase, signed correlation r, and coherence.

    Phase is taken from the complex cross-spectrum summed over each shell and
    unwrapped across k_perp (suppresses +/-pi wrapping of noisy shells); the
    signed r = Re(cross)/sqrt(P_rr P_vv) distinguishes down-gradient (r<0 on
    the up-gradient side) transport from mere coherence.
    """
    cross = s_re + 1j * s_im
    power = np.sqrt(np.maximum(s_rr, 0.0) * np.maximum(s_vv, 0.0))
    denom = np.maximum(s_rr * s_vv, 1e-60)
    coh = (np.abs(cross)**2) / denom
    phase = np.full(cross.shape, np.nan)
    r = np.full(cross.shape, np.nan)
    finite = np.isfinite(power) & (power > 0)
    if np.any(finite):
        pcut = power_frac * np.nanmax(power[finite])
        keepm = finite & (power >= pcut) & (coh >= coh_min)
        idx = np.where(keepm)[0]
        if idx.size >= 2:
            phase[idx] = np.unwrap(np.angle(cross[idx]))
        elif idx.size == 1:
            phase[idx] = np.angle(cross[idx])
        den = np.sqrt(np.maximum(s_rr * s_vv, 1e-60))
        r[idx] = np.real(cross[idx]) / den[idx]
    return phase, r, coh, power


def reduce_spectra(cat, all_data, target_arho=0.05, n_times=6, nx_planes=8):
    """Compute shell-binned cross-spectra for one representative run per gid.

    The representative run is the one with A_rho closest to `target_arho`.
    Results are keyed by gid and contain k/k_perp0, times, and (Nt_sel, Nk)
    arrays of coherence, phase, and signed r — everything F17/F18 need, so
    the plot stage never reopens a snapshot.
    """
    # pick representative tag per gid
    rep = {}
    for tag, run in all_data.items():
        gid = run['meta']['gid']; A = run['meta']['A_rho']
        if A <= 0:
            continue
        if gid not in rep or abs(A - target_arho) < abs(rep[gid][1] - target_arho):
            rep[gid] = (tag, A)

    spectra = {}
    for gid, (tag, A) in sorted(rep.items()):
        snaps = cat[tag]['snapshots']
        run = all_data[tag]; t_arr = run['t']; NT = len(snaps)
        if NT == 0:
            continue
        idx = np.linspace(0, NT - 1, min(n_times, NT), dtype=int)
        try:
            d0 = aread.athdf(snaps[0])
            rho0 = np.asarray(d0['rho'], dtype=np.float64)
            NZ, NY, NX = rho0.shape
            Ly = float(d0['RootGridX2'][1]) - float(d0['RootGridX2'][0])
            Lz = float(d0['RootGridX3'][1]) - float(d0['RootGridX3'][0])
            del rho0, d0; gc.collect()
        except Exception as exc:
            print(f'  WARN spectra {tag}: {exc}'); continue

        k_min = 2 * np.pi / max(Ly, Lz)
        k_max = np.pi * min(NY / Ly, NZ / Lz)
        N_bins = max(40, min(NY // 4, 80))
        edges = np.logspace(np.log10(k_min * 0.9), np.log10(k_max * 1.1),
                            N_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

        coh_ts, phase_ts, r_ts, times = [], [], [], []
        for si in idx:
            try:
                raw = aread.athdf(snaps[si])
            except Exception:
                continue
            rho3d = np.asarray(raw['rho'], dtype=np.float64)
            vy3d = np.asarray(raw['vel2'], dtype=np.float64)
            del raw; gc.collect()
            r0 = np.mean(rho3d, axis=(0, 2), keepdims=True)
            dvy = vy3d - np.mean(vy3d, axis=(0, 2), keepdims=True)
            drho = (rho3d - r0) / (r0 + 1e-30)
            del rho3d, vy3d; gc.collect()
            ixs = np.linspace(0, NX - 1, min(nx_planes, NX), dtype=int)
            acc = [np.zeros(N_bins) for _ in range(4)]
            cnt = np.zeros(N_bins, dtype=np.int64)
            for ix in ixs:
                rr, vv, re, im, c = _spec_accumulators(
                    drho[:, :, ix], dvy[:, :, ix], Ly, Lz, edges)
                acc[0] += rr; acc[1] += vv; acc[2] += re; acc[3] += im
                cnt += c
            del drho, dvy; gc.collect()
            cs = np.maximum(cnt, 1)
            Prr, Pvv = acc[0] / cs, acc[1] / cs
            Pre, Pim = acc[2] / cs, acc[3] / cs
            coh = (Pre**2 + Pim**2) / np.maximum(Prr * Pvv, 1e-60)
            phase, r_sh, _, _ = _weighted_shell_phase_and_r(Prr, Pvv, Pre, Pim)
            coh_ts.append(coh); phase_ts.append(phase); r_ts.append(r_sh)
            times.append(float(t_arr[si]) if si < len(t_arr) else np.nan)

        if not coh_ts:
            continue
        spectra[gid] = dict(
            tag=tag, A_rho=float(A),
            k_over_kp0=centres / run['meta']['k_perp_phys'],
            kdiss_over_kp0=float(k_diss_of(gid) / run['meta']['k_perp_phys']),
            times=np.asarray(times),
            coh=np.vstack(coh_ts), phase=np.vstack(phase_ts),
            r=np.vstack(r_ts),
        )
        print(f'  spectra: gid={gid} ({tag})  {len(times)} times, '
              f'{centres.size} shells')
    return spectra


def save_spectra(spectra, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    flat, meta = {}, {}
    for gid, sp in spectra.items():
        meta[str(gid)] = dict(tag=sp['tag'], A_rho=sp['A_rho'],
                              kdiss_over_kp0=sp['kdiss_over_kp0'])
        for key in ('k_over_kp0', 'times', 'coh', 'phase', 'r'):
            flat[f'g{gid}__{key}'] = sp[key]
    np.savez_compressed(os.path.join(out_dir, SPECTRA_NPZ), **flat)
    with open(os.path.join(out_dir, SPECTRA_JSON), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  -> spectra data : {os.path.join(out_dir, SPECTRA_NPZ)}')


def load_spectra(out_dir=OUTPUT_DIR):
    npz_path = os.path.join(out_dir, SPECTRA_NPZ)
    json_path = os.path.join(out_dir, SPECTRA_JSON)
    if not (os.path.isfile(npz_path) and os.path.isfile(json_path)):
        return None
    with open(json_path) as f:
        meta = json.load(f)
    z = np.load(npz_path)
    spectra = {}
    for gid_s, m in meta.items():
        gid = int(gid_s)
        spectra[gid] = dict(
            tag=m['tag'], A_rho=float(m['A_rho']),
            kdiss_over_kp0=float(m['kdiss_over_kp0']),
            k_over_kp0=z[f'g{gid}__k_over_kp0'], times=z[f'g{gid}__times'],
            coh=z[f'g{gid}__coh'], phase=z[f'g{gid}__phase'],
            r=z[f'g{gid}__r'])
    return spectra


# =============================================================================
#  SECTION 8 — BRIDGE TABLE
# =============================================================================

def compute_bridge_row(tag, run):
    """One summary row per gradient run: chi_A (both definitions), sigma_c,
    eta_eff, anisotropy, K_perp/k_diss, |r|, and the flux-regression alpha."""
    m = run['meta']
    if m['A_rho'] == 0:
        return None
    fit = regress_alpha_from_flux(run)
    lpy = np.where(run['l_perp_y'] > 0, run['l_perp_y'], np.nan)
    lpz = np.where(run['l_perp_z'] > 0, run['l_perp_z'], np.nan)
    chi_box  = chi_A_mean(run, dict(use_measured_lpar=False, use_mean_vA=False))
    chi_meas = chi_A_mean(run, dict(use_measured_lpar=True,  use_mean_vA=True))
    eta_mean = float(np.nanmean(fit['alpha_fit'] * run['zplus_rms'] * lpy))
    return dict(
        tag=tag, gid=int(m['gid']), regime=m['regime'],
        square=bool(m['square']), beta=float(m['beta']),
        A_rho=float(m['A_rho']),
        chi_A_box=chi_box, chi_A_meas=chi_meas,
        sigma_c_mean=float(np.nanmean(run['sigma_c'])),
        eta_eff_mean=eta_mean,
        aniso_y_over_z=float(np.nanmean(lpy / lpz)),
        Kperp_over_kdiss=float(np.nanmean(
            Kperp_phase_mixing(run) / m['k_diss'])),
        rho_vy_corr_rms=float(np.sqrt(np.nanmean(run['rho_vy_corr']**2))),
        alpha_fit=fit['alpha_fit'], alpha_err=fit['alpha_err'],
        alpha_err_wls=fit['alpha_err_wls'],
        residual_rms=fit['residual_rms'], R2=fit['r_squared'],
        n_used=fit['n_used'], status=fit['status'],
    )


def build_bridge_table(all_data):
    rows = []
    for tag in sorted(all_data):
        r = compute_bridge_row(tag, all_data[tag])
        if r is not None:
            rows.append(r)
    return rows


def save_bridge_table_csv(rows, path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, 'bridge_table.csv')
    keys = ['tag', 'regime', 'square', 'beta', 'gid', 'A_rho',
            'chi_A_box', 'chi_A_meas', 'sigma_c_mean', 'eta_eff_mean',
            'aniso_y_over_z', 'Kperp_over_kdiss', 'rho_vy_corr_rms',
            'alpha_fit', 'alpha_err', 'alpha_err_wls', 'residual_rms',
            'R2', 'n_used', 'status']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  -> bridge table CSV : {path}')


def save_bridge_table_latex(rows, path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, 'bridge_table.tex')
    by = {'strong': [], 'weak': [], 'reference': []}
    for r in rows:
        by[r['regime']].append(r)
    L = [r'\begin{table*}', r'\centering',
         (r'\caption{Per-run flux-calibrated transport summary. '
          r'$\alpha$ is the direct flux-regression coefficient of '
          r'$\Gamma_\rho=-\alpha\,z^+_{\rm rms}\ell_{\perp,y}\partial_y\rho_0$ '
          r'with bootstrap $1\sigma$ uncertainty. $\langle\chi_A\rangle$ uses '
          r'the box-mode $\ell_\parallel$ and $v_A=1$; '
          r'$\langle\chi_A\rangle_{\rm m}$ uses the measured parallel ACF '
          r'scale and $\langle v_A\rangle$.}'),
         r'\label{tab:bridge}', r'\begin{tabular}{l c c c c c c c}',
         r'\hline\hline',
         (r'Run & $A_\rho$ & $\langle\chi_A\rangle$ & '
          r'$\langle\chi_A\rangle_{\rm m}$ & $\langle\sigma_c\rangle$ & '
          r'$\langle\ell_{\perp,y}/\ell_{\perp,z}\rangle$ & '
          r'$R^2$ & $\alpha_{\rm eff}$ \\'), r'\hline']
    for reg in ('strong', 'weak', 'reference'):
        for r in by[reg]:
            tg = r['tag'].replace('_', r'\_')
            L.append(
                f'{tg} & {r["A_rho"]:.3g} & {r["chi_A_box"]:.2f} & '
                f'{r["chi_A_meas"]:.2f} & {r["sigma_c_mean"]:.3f} & '
                f'{r["aniso_y_over_z"]:.2f} & {r["R2"]:.3f} & '
                f'${r["alpha_fit"]:.3f}\\pm{r["alpha_err"]:.3f}$ \\\\')
        if by[reg]:
            L.append(r'\hline')
    L += [r'\hline', r'\end{tabular}', r'\end{table*}']
    with open(path, 'w') as f:
        f.write('\n'.join(L))
    print(f'  -> bridge table TeX : {path}')


# =============================================================================
#  SECTION 9 — PLOTTING CONFIGURATION  (edit HERE for Overleaf matching)
# =============================================================================
#  All figure geometry and typography lives in the two dicts below plus
#  FIGSIZES.  Because Stage 2 never touches snapshots, any change here can be
#  re-rendered in seconds with `python ... plot`.
# =============================================================================

MANUSCRIPT_RC = {
    # Typography tuned for aastex631 two-column at \includegraphics widths.
    'font.family':         'serif',
    'font.serif':          ['Computer Modern Roman', 'Times New Roman',
                            'DejaVu Serif'],
    'mathtext.fontset':    'cm',
    'font.size':           15,
    'axes.labelsize':      17,
    'axes.titlesize':      16,
    'legend.fontsize':     13,
    'legend.framealpha':   0.92,
    'legend.edgecolor':    '0.45',
    'xtick.labelsize':     13,
    'ytick.labelsize':     13,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size':    5.5,  'xtick.minor.size': 3.2,
    'ytick.major.size':    5.5,  'ytick.minor.size': 3.2,
    'axes.linewidth':      1.2,
    'lines.linewidth':     2.2,
    'grid.alpha':          0.25,
    'grid.linewidth':      0.7,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.08,
    'figure.dpi':          120,
}

POSTER_RC = {
    'font.size': 22, 'axes.labelsize': 26, 'axes.titlesize': 22,
    'axes.titleweight': 'bold', 'legend.fontsize': 16,
    'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'axes.linewidth': 2.0, 'lines.linewidth': 3.0,
    'xtick.major.size': 8.0, 'ytick.major.size': 8.0,
    'xtick.major.width': 1.8, 'ytick.major.width': 1.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
}

FIGSIZES = {
    'single':    (8.8, 6.0),      # one figure*, single-panel
    'wide':      (11.0, 6.2),     # overlay time-series
    'square':    (9.5, 8.0),      # flux-law collapse
    'chi_panel': (9.2, 6.4),      # master curve
    'grid_1x3':  (17.0, 5.2),
    'grid_2x2':  (12.5, 9.5),
    'grid_2x4':  (19.5, 8.8),
    'grid_2x4t': (20.0, 9.5),
    'poster24':  (22.0, 11.0),
}


def setup_mpl(style='manuscript'):
    """Apply the global rcParams (call once per stage / style switch)."""
    mpl.rcParams.update(MANUSCRIPT_RC)
    if style == 'poster':
        mpl.rcParams.update(POSTER_RC)
    try:
        import subprocess
        subprocess.run(['latex', '--version'], capture_output=True, check=True)
        mpl.rcParams['text.usetex'] = True
    except Exception:
        mpl.rcParams['text.usetex'] = False


# --- colour / marker schemes ------------------------------------------------------

COLOR_ARHO = {0.000: '#1f77b4', 0.025: '#e377c2', 0.050: '#2ca02c',
              0.100: '#d62728', 0.250: '#b8860b', 0.500: '#9467bd'}
GID_COLOR = {0: '#d62728', 1: '#ff7f0e', 2: '#8c564b', 3: '#e377c2',
             4: '#1f77b4', 5: '#17becf', 6: '#9467bd', 7: '#2ca02c',
             8: '#7f7f7f'}
GID_LS = {0: '-', 1: '--', 2: '-.', 3: (0, (5, 2)), 4: ':',
          5: (0, (3, 1, 1, 1)), 6: (0, (1, 1, 5, 1)),
          7: (0, (5, 1, 1, 1)), 8: (0, (4, 2, 1, 2))}
GID_LW = {g: (2.6 if g == 8 else 2.2 if GROUP_META[g].square else 2.0)
          for g in GROUP_META}
REGIME_COLOR = {'strong': '#c0392b', 'weak': '#2471a3', 'reference': '#444444'}
REGIME_BADGE = {
    'strong':    dict(text='STRONG', bg='#ffe6e6', fg='#a30000'),
    'weak':      dict(text='WEAK',   bg='#e6f0ff', fg='#003a99'),
    'reference': dict(text='REF',    bg='#ececec', fg='#333333'),
}


def _color(A_rho):
    return COLOR_ARHO.get(A_rho, '#888888')


def _tick_both(ax):
    ax.tick_params(which='both', direction='in', top=True, right=True)


def savefig(fig, name, dpi=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi) if dpi else fig.savefig(path)
    plt.close(fig)
    print(f'  -> {path}')


def _add_regime_badge(ax, gid, loc='upper left'):
    b = REGIME_BADGE[regime_of(gid)]
    pos = {'upper left':  (0.025, 0.965, 'left',  'top'),
           'upper right': (0.975, 0.965, 'right', 'top'),
           'lower left':  (0.025, 0.035, 'left',  'bottom'),
           'lower right': (0.975, 0.035, 'right', 'bottom')}[loc]
    txt = b['text'] + (' SQ' if is_square(gid) else '')
    ax.text(pos[0], pos[1], txt, transform=ax.transAxes, ha=pos[2], va=pos[3],
            fontsize=11, fontweight='bold', color=b['fg'],
            bbox=dict(boxstyle='round,pad=0.25', facecolor=b['bg'],
                      edgecolor=b['fg'], linewidth=0.8, alpha=0.92))


def _legend_handles(all_data, include_reference=True):
    a_vals = sorted({d['meta']['A_rho'] for d in all_data.values()})
    c = [Patch(facecolor=_color(a), label=rf'$A_\rho={a:.3g}$') for a in a_vals]
    present = {d['meta']['gid'] for d in all_data.values()}
    s = []
    for g in ACTIVE_GIDS_ALL:
        if g not in present:
            continue
        if (not include_reference) and is_reference(g):
            continue
        gm = GROUP_META[g]
        s.append(Line2D([0], [0], color='k', lw=gm.lw, ls=gm.ls,
                        label=gm.label))
    return c, s


def _legend_handles_by_gid(all_data, include_reference=True):
    present = sorted({d['meta']['gid'] for d in all_data.values()})
    h = []
    for g in present:
        if (not include_reference) and is_reference(g):
            continue
        h.append(Line2D([0], [0], color=GID_COLOR[g], lw=GID_LW[g],
                        ls=GID_LS[g], label=GROUP_META[g].label))
    return h


def _make_2x4(figsize=None, sharex=False, sharey=False):
    fig, axes = plt.subplots(2, 4, figsize=figsize or FIGSIZES['grid_2x4'],
                             sharex=sharex, sharey=sharey,
                             constrained_layout=True)
    return fig, axes.flatten()


def marker_for(gid):
    """Regime/profile scatter style: strong=circle, weak=square,
    square-wave=open symbol, reference=star."""
    if is_reference(gid):
        return dict(marker='*', open=False,
                    color=REGIME_COLOR['reference'], size=320)
    shape = 'o' if is_strong(gid) else 's'
    return dict(marker=shape, open=is_square(gid),
                color=REGIME_COLOR[regime_of(gid)], size=150)


def _scatter_run(ax, x, y, gid, size=None, alpha=0.95, label=None, zorder=5):
    info = marker_for(gid); s = size if size else info['size']
    if info['open']:
        ax.scatter(x, y, s=s, marker=info['marker'], facecolors='none',
                   edgecolors=info['color'], linewidths=1.8, alpha=alpha,
                   label=label, zorder=zorder)
    else:
        ax.scatter(x, y, s=s, marker=info['marker'], color=info['color'],
                   edgecolors='k', linewidths=0.7, alpha=alpha,
                   label=label, zorder=zorder)


def _regime_profile_legend():
    return [
        Line2D([0], [0], marker='o', color=REGIME_COLOR['strong'], lw=0,
               markeredgecolor='k', markersize=12,
               label=r'Strong (sinusoidal)'),
        Line2D([0], [0], marker='s', color=REGIME_COLOR['weak'], lw=0,
               markeredgecolor='k', markersize=11, label=r'Weak (sinusoidal)'),
        Line2D([0], [0], marker='o', color='none', lw=0,
               markeredgecolor=REGIME_COLOR['strong'], markersize=12,
               markeredgewidth=1.8, label=r'Strong (square)'),
        Line2D([0], [0], marker='s', color='none', lw=0,
               markeredgecolor=REGIME_COLOR['weak'], markersize=11,
               markeredgewidth=1.8, label=r'Weak (square)'),
        Line2D([0], [0], marker='*', color=REGIME_COLOR['reference'], lw=0,
               markeredgecolor='k', markersize=16, label=r'Reference'),
    ]


def _representative_run(all_data, gid):
    """Highest-A_rho gradient run of a family (for single-run panels)."""
    runs = [(tag, d) for tag, d in all_data.items()
            if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0]
    if not runs:
        return None, None
    return sorted(runs, key=lambda kv: kv[1]['meta']['A_rho'])[-1]


def _runs_of(all_data, gid, gradients_only=True):
    runs = [(tag, d) for tag, d in all_data.items()
            if d['meta']['gid'] == gid
            and (d['meta']['A_rho'] > 0 or not gradients_only)]
    return sorted(runs, key=lambda kv: kv[1]['meta']['A_rho'])


# =============================================================================
#  SECTION 10 — FIGURES  (one independent function per figure)
# =============================================================================

def fig_zplus_overlay(all_data, report=None):
    """F01: z^+(t)/z^+(0) overlay, gid-keyed styles; dissipation-onset marks."""
    fig, ax = plt.subplots(figsize=FIGSIZES['wide'], constrained_layout=True)
    for tag, d in all_data.items():
        gid = d['meta']['gid']; t = d['t']; zp = d['zplus_rms']
        mk = t <= T_MAX_PLOT; tp, zpp = t[mk], zp[mk]
        zp0 = zpp[0] if len(zpp) and zpp[0] > 1e-12 else 1.0
        ax.semilogy(tp, zpp / zp0, color=GID_COLOR[gid], ls=GID_LS[gid],
                    lw=GID_LW[gid], alpha=0.92)
        t_d = t_diss_inst(gid, d['dvAdy_max'][0])
        if np.isfinite(t_d) and 0 < t_d <= T_MAX_PLOT:
            ax.axvline(t_d, color=GID_COLOR[gid], ls=GID_LS[gid],
                       lw=0.9, alpha=0.30)
    ax.set_xlim(0, T_MAX_PLOT)
    ax.set_xlabel(r'$t/t_A$')
    ax.set_ylabel(r'$z^+_{\rm rms}(t)/z^+_{\rm rms}(0)$')
    _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    ax.legend(handles=_legend_handles_by_gid(all_data), loc='lower left',
              framealpha=0.92, ncol=2)
    savefig(fig, 'F01_zplus_overlay.pdf')


def _waterfall_panel(ax, d, gid):
    t = d['t']; t_max = t.max() if t.size else 1.0
    norm = Normalize(vmin=0, vmax=t_max); cmap = plt.cm.viridis
    stride = max(1, len(t) // 45)
    for i in range(0, len(t), stride):
        ax.plot(d['y_arr'], d['rho0'][i], color=cmap(norm(t[i])),
                lw=0.9, alpha=0.70)
    ax.plot(d['y_arr'], d['rho0'][0], color='k', lw=2.8, zorder=6,
            label=r'$t=0$')
    ax.set_xlabel(r'$y$'); ax.set_ylabel(r'$\langle\rho\rangle_{xz}(y,t)$')
    ax.set_title(panel_title(gid, d['meta']['A_rho']))
    _tick_both(ax); ax.grid(True, alpha=0.20); ax.legend(loc='best')
    _add_regime_badge(ax, gid, loc='upper right')
    return norm, cmap


def fig_rho_waterfalls(all_data, report=None):
    """F02: representative waterfalls, 2x2 = S1, S4(sq) / W1, W4(sq)."""
    order = [0, 3, 4, 7]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['grid_2x2'],
                             constrained_layout=True)
    axes = axes.flatten(); last = None
    for ax, gid in zip(axes, order):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        last = _waterfall_panel(ax, d, gid)
    if last:
        sm = ScalarMappable(cmap=last[1], norm=last[0]); sm.set_array([])
        cb = fig.colorbar(sm, ax=list(axes), location='right',
                          fraction=0.02, pad=0.01)
        cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02_rho_waterfalls_representative.pdf')


def fig_reference_waterfall(all_data, report=None):
    """F02b: standalone waterfall for the reference (chi_A ~ 1) run."""
    refs = [d for d in all_data.values()
            if is_reference(d['meta']['gid']) and d['meta']['A_rho'] > 0]
    if not refs:
        print('  [skip F02b] no reference runs'); return
    d = sorted(refs, key=lambda r: -r['meta']['A_rho'])[0]
    gid = d['meta']['gid']
    fig, ax = plt.subplots(figsize=FIGSIZES['single'], constrained_layout=True)
    norm, cmap = _waterfall_panel(ax, d, gid)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02b_reference_waterfall.pdf')


def fig_rho_waterfalls_full(all_data, report=None):
    """F02c: full-suite waterfalls, 2x4 = S1..S4 / W1..W4."""
    fig, axes = _make_2x4(figsize=FIGSIZES['grid_2x4t']); last = None
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        last = _waterfall_panel(ax, d, gid)
    if last:
        sm = ScalarMappable(cmap=last[1], norm=last[0]); sm.set_array([])
        cb = fig.colorbar(sm, ax=list(axes), location='right',
                          fraction=0.015, pad=0.01)
        cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02c_rho_waterfalls_full_2x4.pdf')


def fig_lperp_comparison(all_data, report=None):
    """F03: l_perp,y / l_perp,z / l_perp,full evolution for every run."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['grid_1x3'],
                             constrained_layout=True)
    keys = ['l_perp_y', 'l_perp_z', 'l_perp_full']
    titles = [r'$\ell_{\perp,y}$', r'$\ell_{\perp,z}$',
              r'$\ell_{\perp,\rm full}$']
    for ax, key, ttl in zip(axes, keys, titles):
        for tag, d in all_data.items():
            gid = d['meta']['gid']; gm = GROUP_META[gid]
            ax.semilogy(d['t'], d[key], color=_color(d['meta']['A_rho']),
                        ls=gm.ls, lw=gm.lw, alpha=0.88)
        ax.set_xlabel(r'$t/t_A$'); ax.set_title(ttl)
        _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel(r'$\ell_\perp$')
    c_h, s_h = _legend_handles(all_data)
    axes[2].legend(handles=c_h + s_h, loc='lower left', fontsize=11, ncol=2)
    savefig(fig, 'F03_lperp_comparison.pdf')


def fig_alpha_dynamic(all_data, report=None):
    """F04: coarse k=1 erosion alpha(t) (consistency observable), 2x4."""
    fig, axes = _make_2x4(sharey=True)
    if report is not None:
        report += ['\n' + '=' * 70,
                   'COARSE EROSION OBSERVABLE  (k=1 alpha consistency check)',
                   '=' * 70]
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            ax.plot(d['t'], coarse_erosion_alpha(d), color=_color(m['A_rho']),
                    lw=2.2, label=rf'$A_\rho={m["A_rho"]:.3g}$')
            if report is not None:
                af, _ = alpha_flux(d)
                report.append(f'  {tag:24s}  [{regime_of(gid):>9s}]  '
                              f'alpha_flux={af:.4f}  '
                              f'chi_A={chi_A_mean(d):.2f}')
        ax.axhline(1 / 8, color='k', ls='--', lw=1.0, alpha=0.5,
                   label=r'$1/8$')
        ax.set_title(panel_title(gid, A_repr))
        ax.set_xlim(left=0); ax.set_ylim(0, 0.42); ax.set_xlabel(r'$t/t_A$')
        _tick_both(ax); ax.grid(True, alpha=0.22); ax.legend(fontsize=11)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$\alpha_{k=1}(t)$')
    axes[4].set_ylabel(r'$\alpha_{k=1}(t)$')
    savefig(fig, 'F04_coarse_alpha_k1.pdf')


def fig_alpha_inst(all_data, report=None):
    """F05: per-snapshot flux-regression alpha_inst(t), 2x4.

    Visual justification of the transient cut and the time-stationarity of
    the transport coefficient (reviewer-facing companion to F09/F09b).
    """
    fig, axes = _make_2x4(sharey=True)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            a_t = regress_alpha_per_time(d)
            ax.plot(d['t'], a_t, color=_color(m['A_rho']), lw=2.0,
                    label=rf'$A_\rho={m["A_rho"]:.3g}$')
            # shade the transient window excluded by the global fit
            n_skip = int(np.ceil(0.05 * len(d['t'])))
            if n_skip > 0 and A_repr == m['A_rho']:
                ax.axvspan(d['t'][0], d['t'][min(n_skip, len(d['t']) - 1)],
                           color='0.75', alpha=0.25, zorder=0)
        ax.axhline(1 / 8, color='k', ls='--', lw=1.0, alpha=0.5,
                   label=r'$1/8$')
        ax.set_title(panel_title(gid, A_repr))
        ax.set_xlim(left=0); ax.set_ylim(-0.05, 0.42)
        ax.set_xlabel(r'$t/t_A$')
        _tick_both(ax); ax.grid(True, alpha=0.22); ax.legend(fontsize=11)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$\alpha_{\rm inst}(t)$')
    axes[4].set_ylabel(r'$\alpha_{\rm inst}(t)$')
    savefig(fig, 'F05_alpha_inst.pdf')


def fig_chi_A(all_data, report=None):
    """F06: chi_A(t) overlay with regime shading."""
    fig, ax = plt.subplots(figsize=FIGSIZES['wide'], constrained_layout=True)
    for tag, d in all_data.items():
        gid = d['meta']['gid']
        ax.semilogy(d['t'], chi_A(d), color=GID_COLOR[gid], ls=GID_LS[gid],
                    lw=GID_LW[gid], alpha=0.92)
    ax.axhspan(1e-3, 1.0, color=REGIME_BADGE['weak']['bg'], alpha=0.45,
               zorder=0)
    ax.axhspan(1.0, 1e2, color=REGIME_BADGE['strong']['bg'], alpha=0.45,
               zorder=0)
    ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel(r'$t/t_A$')
    ax.set_ylabel(
        r'$\chi_A=z^+_{\rm rms}\,\ell_{\parallel,0}/(v_A\,\ell_{\perp,y})$')
    ax.set_xlim(left=0); ax.set_ylim(3e-2, 1e2)
    _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    handles = _legend_handles_by_gid(all_data) + [
        Patch(facecolor=REGIME_BADGE['weak']['bg'], alpha=0.55,
              label=r'$\chi_A<1$'),
        Patch(facecolor=REGIME_BADGE['strong']['bg'], alpha=0.55,
              label=r'$\chi_A>1$')]
    ax.legend(handles=handles, loc='upper right', framealpha=0.92, ncol=2)
    if report is not None:
        report += ['\n' + '=' * 70, 'CHI_A — time-averaged (box / measured)',
                   '=' * 70]
        for tag, d in sorted(all_data.items()):
            if d['meta']['A_rho'] == 0:
                continue
            cb = chi_A_mean(d, dict(use_measured_lpar=False,
                                    use_mean_vA=False))
            cm = chi_A_mean(d, dict(use_measured_lpar=True, use_mean_vA=True))
            report.append(f'  {tag:24s}  <chi_A>_box={cb:5.2f}  '
                          f'<chi_A>_meas={cm:5.2f}')
    savefig(fig, 'F06_chi_A.pdf')


def fig_flux_collapse(all_data, report=None):
    """F09: single-panel flux-law collapse of the whole suite."""
    fig, ax = plt.subplots(figsize=FIGSIZES['square'], constrained_layout=True)
    if report is not None:
        report += ['\n' + '=' * 70,
                   'DIRECT FLUX-LAW COLLAPSE  (single panel)', '=' * 70]
    rng = np.random.default_rng(7)
    all_X = []
    for tag, d in sorted(all_data.items(),
                         key=lambda kv: kv[1]['meta']['gid']):
        m = d['meta']
        if m['A_rho'] == 0:
            continue
        fit = regress_alpha_from_flux(d)
        if fit['status'] != 'ok':
            if report is not None:
                report.append(f'  {tag:24s}  status={fit["status"]}')
            continue
        X = fit['X'][fit['mask']].ravel(); Y = fit['Y'][fit['mask']].ravel()
        all_X.append(X)
        if X.size > 400:
            idx = rng.choice(X.size, 400, replace=False)
            X, Y = X[idx], Y[idx]
        _scatter_run(ax, X, Y, m['gid'], size=22, alpha=0.30, zorder=3)
        if report is not None:
            report.append(
                f'  {tag:24s}  [{regime_of(m["gid"]):>9s}]  '
                f'alpha={fit["alpha_fit"]:.4f} +/- {fit["alpha_err"]:.4f}  '
                f'R2={fit["r_squared"]:.3f}  n={fit["n_used"]}')
    if all_X:
        Xc = np.concatenate(all_X)
        xs = np.linspace(np.nanpercentile(Xc, 1), np.nanpercentile(Xc, 99),
                         100)
        ax.plot(xs, (1 / 8.) * xs, 'k--', lw=2.2, zorder=6)
    ax.axhline(0, color='0.6', lw=0.7); ax.axvline(0, color='0.6', lw=0.7)
    ax.set_xlabel(r'$-z^+_{\rm rms}\,\ell_{\perp,y}\,\partial_y\rho_0$')
    ax.set_ylabel(r'$\Gamma_\rho=\langle\delta\rho\,\delta v_y\rangle_{xz}$')
    _tick_both(ax); ax.grid(True, alpha=0.20)
    leg = ax.legend(handles=_regime_profile_legend() +
                    [Line2D([0], [0], color='k', ls='--', lw=2.2,
                            label=r'$\alpha_{\rm th}=1/8$')],
                    loc='upper left', framealpha=0.95)
    leg.set_zorder(10)
    savefig(fig, 'F09_flux_law_collapse.pdf')


def fig_flux_regression_clouds(all_data, report=None):
    """F09b: per-family hexbin regression clouds with fitted slopes, 2x4."""
    fig, axes = _make_2x4(figsize=FIGSIZES['grid_2x4t'])
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        fit = regress_alpha_from_flux(d)
        if fit['status'] != 'ok':
            ax.axis('off'); continue
        Xk = fit['X'][fit['mask']]; Yk = fit['Y'][fit['mask']]
        ax.hexbin(Xk, Yk, gridsize=38, cmap='viridis', mincnt=1, linewidths=0)
        xs = np.linspace(Xk.min(), Xk.max(), 80)
        ax.plot(xs, fit['alpha_fit'] * xs, color='#d62728', lw=2.6,
                label=(rf'$\alpha={fit["alpha_fit"]:.3f}'
                       rf'\pm{fit["alpha_err"]:.3f}$'))
        a_th = alpha_th_imbalance(float(np.nanmean(d['sigma_c'])))
        ax.plot(xs, a_th * xs, 'k--', lw=1.6, alpha=0.8,
                label=rf'$\alpha_{{\rm th}}={a_th:.3f}$')
        ax.axhline(0, color='0.6', lw=0.7); ax.axvline(0, color='0.6', lw=0.7)
        ax.set_xlabel(r'$-z^+_{\rm rms}\,\ell_{\perp,y}\,\partial_y\rho_0$')
        ax.set_ylabel(r'$\Gamma_\rho$')
        ax.set_title(panel_title(gid, d['meta']['A_rho']))
        _tick_both(ax); ax.grid(True, alpha=0.18)
        ax.text(0.03, 0.97, rf'$R^2={fit["r_squared"]:.3f}$',
                transform=ax.transAxes, ha='left', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.6', alpha=0.9))
        ax.legend(loc='lower right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper right')
    savefig(fig, 'F09b_flux_regression_clouds.pdf')


def fig_alpha_vs_chi(all_data, report=None):
    """F10: master curve — alpha_eff vs <chi_A> with the Lorentzian bridge."""
    fig, ax = plt.subplots(figsize=FIGSIZES['chi_panel'],
                           constrained_layout=True)
    if report is not None:
        report += ['\n' + '=' * 70,
                   'ALPHA_FLUX vs CHI_A  (Lorentzian theory)', '=' * 70]
    ax.axvspan(1e-2, 0.7, color=REGIME_BADGE['weak']['bg'], alpha=0.35,
               zorder=0)
    ax.axvspan(1.5, 1e2, color=REGIME_BADGE['strong']['bg'], alpha=0.35,
               zorder=0)
    cg = np.logspace(-1.5, 1.7, 200)
    ax.plot(cg, np.full_like(cg, 1 / 8.), 'k--', lw=1.4, alpha=0.6)
    ax.plot(cg, lorentzian_alpha(cg), 'k-', lw=2.0, alpha=0.9)
    ax.axvline(1.0, color='k', ls=':', lw=0.9, alpha=0.5)
    for tag, d in all_data.items():
        m = d['meta']
        if m['A_rho'] == 0:
            continue
        fit = regress_alpha_from_flux(d)
        af = fit['alpha_fit']
        cm = chi_A_mean(d)
        if not np.isfinite(af):
            continue
        _scatter_run(ax, cm, af, m['gid'])
        ax.errorbar(cm, af, yerr=fit['alpha_err'], fmt='none',
                    ecolor='0.35', elinewidth=1.0, capsize=2.5, zorder=4)
        if report is not None:
            ap = float(lorentzian_alpha(cm))
            re = (af - ap) / ap if ap > 0 else np.nan
            report.append(
                f'  {tag:24s}  [{regime_of(m["gid"]):>9s}]  chi_A={cm:5.2f}  '
                f'alpha={af:.4f}+/-{fit["alpha_err"]:.4f}  pred={ap:.4f}  '
                f'rel_err={re:+.1%}')
    ax.set_xscale('log'); ax.set_xlabel(r'$\langle\chi_A\rangle$')
    ax.set_ylabel(r'$\alpha_{\rm eff}$')
    ax.set_ylim(0, 0.30); ax.set_xlim(0.1, 30)
    _tick_both(ax); ax.grid(True, alpha=0.22)
    th = [Line2D([0], [0], color='k', ls='-', lw=2.0,
                 label=r'$\alpha_{\rm eff}=\tfrac18\chi_A^2/(1+\chi_A^2)$'),
          Line2D([0], [0], color='k', ls='--', lw=1.4, label=r'$1/8$')]
    ax.legend(handles=_regime_profile_legend() + th, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=12)
    savefig(fig, 'F10_alpha_vs_chi.pdf')


def fig_morphology_diagnostics(all_data, report=None):
    """F10b: variance + k=1..4 harmonic amplitudes; delta_grad secondary."""
    fig, axes = _make_2x4(figsize=FIGSIZES['grid_2x4t'], sharey=False)
    LS_K = {2: '--', 3: '-.', 4: ':'}
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        mo = compute_morphology(d); t = mo['t']
        A1_0 = mo['A_modes'][1][0] if mo['A_modes'][1][0] > 1e-14 else 1.0
        var_norm = mo['var'] / max(mo['var'][0], 1e-30)
        ax.semilogy(t, np.maximum(np.sqrt(var_norm), 1e-4), color='k', lw=2.8,
                    zorder=6, label=r'$\sqrt{\mathrm{Var}/\mathrm{Var}_0}$')
        ax.semilogy(t, np.maximum(mo['A_modes'][1] / A1_0, 1e-4),
                    color=_color(d['meta']['A_rho']), lw=2.6, zorder=5,
                    label=r'$k{=}1$')
        for k in (2, 3, 4):
            ax.semilogy(t, np.maximum(mo['A_modes'][k] / A1_0, 1e-4),
                        color=_color(d['meta']['A_rho']), ls=LS_K[k], lw=1.8,
                        alpha=0.9, zorder=4, label=rf'$k{{=}}{k}$')
        dg = mo['delta_grad']
        dg0 = dg[0] if np.isfinite(dg[0]) and dg[0] > 0 else 1.0
        ax.semilogy(t, np.maximum(dg / dg0, 1e-4), color='0.55', lw=1.2,
                    ls=(0, (1, 1)), alpha=0.45, zorder=2,
                    label=r'$\delta_{\rm grad}$ (sec.)')
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'normalised amplitude')
        ax.set_xlim(left=0); ax.set_ylim(1e-3, 2.0)
        ax.set_title(panel_title(gid, d['meta']['A_rho']))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        _add_regime_badge(ax, gid, loc='upper right')
    axes[0].legend(loc='lower left', fontsize=11, framealpha=0.92, ncol=2)
    savefig(fig, 'F10b_morphology.pdf')


def fig_gradient_thickness(all_data, report=None):
    """F10c: interface thickness delta_grad(t); square families emphasised."""
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        emphasise = is_square(gid); A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            mo = compute_morphology(d)
            col = _color(m['A_rho']) if emphasise else '0.55'
            ax.plot(mo['t'], mo['delta_grad'], color=col,
                    lw=2.6 if emphasise else 1.4,
                    alpha=0.95 if emphasise else 0.6,
                    label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$')
        ax.set_ylabel(
            r'$\delta_{\rm grad}=\Delta\rho_0/\max|\partial_y\rho_0|$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        if emphasise:
            ax.legend(loc='best', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F10c_gradient_thickness.pdf')


def fig_erosion_raw(all_data, report=None):
    """F11: raw k=1 erosion amplitude (consistency check), 2x4."""
    fig, axes = _make_2x4(sharey=True)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            ax.semilogy(d['t'], d['A_rho_norm'], color=_color(m['A_rho']),
                        lw=2.5, label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylim(1e-3, 2.0); ax.set_xlim(left=0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='lower left', fontsize=11, framealpha=0.90)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$A_\rho(t)/A_\rho(0)$')
    axes[4].set_ylabel(r'$A_\rho(t)/A_\rho(0)$')
    savefig(fig, 'F11_k1_erosion_check.pdf')


def compute_F11b_data(all_data):
    """Derive the F11b prediction/measurement bundle from reduced arrays.

    Cheap (no snapshot access); kept as a separate function so the bundle
    can be inspected or exported independently of the plot.
    """
    F11b = {}
    for tag, d in all_data.items():
        m = d['meta']
        if m['A_rho'] == 0:
            continue
        sm = float(np.nanmean(d['sigma_c']))
        a_th = alpha_th_imbalance(sm)
        a_fx, _ = alpha_flux(d)
        reg = regime_of(m['gid'])
        modes = {}
        for n in (1, 2):
            md = dict(A_meas=(d['A_rho_norm'] if n == 1
                              else d['A_rho_k2_norm']))
            if reg == 'strong':
                _, md['P_const'], _ = build_prediction(d, a_th, mode_n=n,
                                                       suppression='none')
                md['P_bf'] = (build_prediction(d, a_fx, mode_n=n,
                                               suppression='none')[1]
                              if np.isfinite(a_fx) else None)
                md['P_chi2'] = None
            else:
                _, md['P_chi2'], _ = build_prediction(d, a_th, mode_n=n,
                                                      suppression='weak')
                md['P_const'] = None; md['P_bf'] = None
            modes[n] = md
        F11b[tag] = dict(gid=int(m['gid']), regime=reg,
                         A_rho=float(m['A_rho']),
                         chi_A_mean=chi_A_mean(d), alpha_th=a_th,
                         alpha_bf=a_fx, t=d['t'], modes=modes)
    return F11b


def fig_theory_vs_measurement(all_data, report=None):
    """F11b: k=1/k=2 erosion vs flux-calibrated predictions (poster style)."""
    if report is not None:
        report += ['\n' + '=' * 70,
                   'F11b k=1/k=2 EROSION  (flux-calibrated alpha)', '=' * 70]
    F11b = compute_F11b_data(all_data)
    if report is not None:
        for tag, rd in sorted(F11b.items()):
            report.append(f'  {tag:24s}  [{rd["regime"]:>9s}]  '
                          f'a_flux={rd["alpha_bf"]:.4f}  '
                          f'<chi_A>={rd["chi_A_mean"]:.2f}')
    setup_mpl('poster')
    A_FLOOR = 2e-3
    fig, axes = _make_2x4(figsize=FIGSIZES['poster24'], sharey=True)
    LS_M1 = '-'; LS_M2 = (0, (6, 2.5)); LS_C = ':'
    LS_B = (0, (3, 1.5, 1, 1.5)); LS_W = (0, (5, 1.5, 1, 1.5))
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(t_, rd) for t_, rd in F11b.items()
                       if rd['gid'] == gid], key=lambda x: x[1]['A_rho'])
        flagged = dict(const=False, bf=False, chi2=False); A_repr = None
        for tag, rd in runs:
            t = rd['t']; col = _color(rd['A_rho']); reg = rd['regime']
            md1, md2 = rd['modes'][1], rd['modes'][2]; A_repr = rd['A_rho']
            ax.semilogy(t, np.abs(md1['A_meas']) + A_FLOOR, color=col,
                        ls=LS_M1, lw=4.0,
                        label=rf'$A_\rho={rd["A_rho"]:.3g}$, $k{{=}}1$')
            ax.semilogy(t, np.abs(md2['A_meas']) + A_FLOOR, color=col,
                        ls=LS_M2, lw=3.0, alpha=0.85,
                        label=rf'$A_\rho={rd["A_rho"]:.3g}$, $k{{=}}2$')
            if reg == 'strong':
                if md1['P_const'] is not None:
                    ax.semilogy(t, md1['P_const'], color=col, ls=LS_C, lw=2.6,
                                alpha=0.8,
                                label=(r'$\alpha_{\rm th}$'
                                       if not flagged['const'] else None))
                    flagged['const'] = True
                if md1['P_bf'] is not None:
                    ax.semilogy(t, md1['P_bf'], color=col, ls=LS_B, lw=2.6,
                                alpha=0.9,
                                label=(rf'$\alpha_{{\rm flux}}'
                                       rf'={rd["alpha_bf"]:.3f}$'
                                       if not flagged['bf'] else None))
                    flagged['bf'] = True
            else:
                if md1['P_chi2'] is not None:
                    ax.semilogy(t, md1['P_chi2'], color=col, ls=LS_W, lw=2.6,
                                alpha=0.9,
                                label=(r'$\alpha_{\rm th}\chi_A^2$'
                                       if not flagged['chi2'] else None))
                    flagged['chi2'] = True
        ax.axhline(0.5, color='0.40', ls=':', lw=1.6, alpha=0.6)
        ax.set_xlabel(r'$t/t_A$'); ax.set_xlim(left=0)
        ax.set_ylim(A_FLOOR * 0.5, 2.0)
        ax.set_title(panel_title(gid, A_repr))
        ax.grid(True, which='both', alpha=0.25); _tick_both(ax)
        ax.legend(loc='lower left', framealpha=0.92, handlelength=2.4,
                  fontsize=13)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$A_\rho(t)/A_{\rho,k=1}(0)$')
    axes[4].set_ylabel(r'$A_\rho(t)/A_{\rho,k=1}(0)$')
    savefig(fig, 'F11b_erosion_modes.pdf', dpi=300)
    setup_mpl('manuscript')


def fig_integrated_flux(all_data, report=None):
    """F12: int |Gamma_rho| dy time series per family, 2x4."""
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            ax.plot(d['t'], integrate_abs_flux(d), color=_color(m['A_rho']),
                    lw=2.4, label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$')
        ax.set_ylabel(r'$\int|\Gamma_\rho|\,\mathrm{d}y$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F12_integrated_flux.pdf')


def fig_slaved_closure(all_data, report=None):
    """F14: measured flux vs fixed- and dynamic-l_perp closure predictions."""
    fig, axes = _make_2x4(figsize=FIGSIZES['grid_2x4'], sharey=False)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        m = d['meta']; t = d['t']
        a_th = alpha_th_imbalance(float(np.nanmean(d['sigma_c'])))
        gamma_meas = np.sqrt(np.nanmean(d['Gamma_rho']**2, axis=1))
        grad_rms = np.sqrt(np.nanmean(d['drho0_dy']**2, axis=1))
        l0 = d['l_perp_y'][0]
        l_fixed = l0 if np.isfinite(l0) and l0 > 0 \
            else 1.0 / max(m['k_perp_phys'], 1e-30)
        pred_fixed = a_th * d['zplus_rms'] * l_fixed * grad_rms
        lpy = np.where(d['l_perp_y'] > 0, d['l_perp_y'], np.nan)
        pred_dyn = a_th * d['zplus_rms'] * lpy * grad_rms
        ax.plot(t, gamma_meas, color='k', lw=2.6,
                label=r'Measured $\Gamma_\rho$')
        ax.plot(t, pred_fixed, color='#1f77b4', ls='--', lw=2.2,
                label=r'Expected (fixed $\ell_\perp$)')
        ax.plot(t, pred_dyn, color='#d62728', ls='-', lw=2.2,
                label=r'Expected (computed $\ell_\perp$)')
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$\Gamma_\rho$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, m['A_rho']))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='best', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F14_slaved_closure.pdf')


def fig_slaved_closure_measured(all_data, report=None):
    """F14b: measured (drho/rho0)_rms only, per family, 2x4."""
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = _runs_of(all_data, gid)
        if not runs:
            ax.axis('off'); continue
        A_repr = runs[-1][1]['meta']['A_rho']
        for tag, d in runs:
            m = d['meta']
            ax.plot(d['t'], d['drho_over_rho0_rms'], color=_color(m['A_rho']),
                    lw=2.6, label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$')
        ax.set_ylabel(r'$(\delta\rho/\rho_0)_{\rm rms}$')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F14b_slaved_closure_measured.pdf')


def fig_Kperp_vs_dissipation(all_data, report=None):
    """F15: phase-mixing K_perp(t) vs k_diss and 1/l_perp,y, 2x4."""
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        ax.axhline(k_diss_of(gid), color='k', ls='--', lw=1.5, alpha=0.7,
                   label=r'$k_{\rm diss}$')
        ax.axhline(GROUP_META[gid].kappa_perp * 2 * np.pi, color='0.5',
                   ls=':', lw=1.2, alpha=0.6, label=r'$k_{\perp,0}$')
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']; col = _color(m['A_rho'])
            ax.semilogy(d['t'], Kperp_phase_mixing(d), color=col, ls='-',
                        lw=2.2)
            ilpy = np.where(d['l_perp_y'] > 0, 1. / d['l_perp_y'], np.nan)
            ax.semilogy(d['t'], ilpy, color=col, ls='-.', lw=1.4, alpha=0.7)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$k_\perp\;[L_y^{-1}]$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='lower right', fontsize=11)
        _add_regime_badge(ax, gid, loc='upper right')
    savefig(fig, 'F15_Kperp_vs_kdiss.pdf')


def fig_slaved_ratio(all_data, report=None):
    """F16: slaved-amplitude ratio R(t).

    R = (l_perp,full F_rho / sqrt(2)) / (drho/rho0)_rms; the 1/sqrt(2) is
    the perpendicular ISOTROPY projection <dv_y^2> = (1/2)<|dv_perp|^2>
    (NOT an amplitude-convention factor; see manuscript Sec. 2.5.3).
    """
    PROJ = 1.0 / np.sqrt(2.0)
    fig, axes = _make_2x4()
    if report is not None:
        report += ['\n' + '=' * 70, 'SLAVED RATIO R(t)', '=' * 70]
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        A_repr = None
        for tag, d in _runs_of(all_data, gid):
            m = d['meta']; A_repr = m['A_rho']
            lps = np.where(d['l_perp_full'] > 0, d['l_perp_full'], 0.0)
            R = PROJ * lps * d['Frho_rms_y'] / (d['drho_over_rho0_rms'] + 1e-30)
            ax.plot(d['t'], R, color=_color(m['A_rho']), lw=2.4,
                    label=rf'$A_\rho={m["A_rho"]:.3g}$')
            if report is not None:
                Rf = R[np.isfinite(R)]
                rmean = float(np.nanmean(Rf)) if Rf.size else np.nan
                report.append(f'  {tag:24s}  <R>={rmean:.3f}')
        ax.axhline(1.0, color='k', ls='--', lw=1.4, alpha=0.65)
        ax.axhspan(0.8, 1.2, color='grey', alpha=0.10)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$R(t)$')
        ax.set_xlim(left=0); ax.set_ylim(0, 3.0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F16_slaved_ratio.pdf')


def fig_cross_coherence(all_data, spectra, report=None):
    """F17: squared coherence C(k_perp, t) from precomputed spectra, 2x4."""
    if spectra is None:
        print('  [skip F17] spectra not reduced; rerun reduce with --spectra')
        return
    fig, axes = _make_2x4()
    tmax = max((sp['times'].max() for sp in spectra.values()), default=1.0)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        if gid not in spectra:
            ax.axis('off'); continue
        sp = spectra[gid]
        norm_t = Normalize(vmin=0, vmax=tmax); cmap_t = plt.cm.viridis
        for i, tv in enumerate(sp['times']):
            ax.plot(sp['k_over_kp0'], sp['coh'][i],
                    color=cmap_t(norm_t(tv)), lw=1.8, alpha=0.92,
                    label=(rf'$t={tv:.1f}$' if i == 0 else None))
        ax.axvline(1.0, color='k', ls='--', lw=1.0, alpha=0.55)
        ax.axvline(sp['kdiss_over_kp0'], color='r', ls='--', lw=1.0,
                   alpha=0.55)
        ax.set_xscale('log'); ax.set_xlabel(r'$k_\perp/k_{\perp,0}$')
        ax.set_ylabel(r'$C(k_\perp)$')
        ax.set_xlim(left=0.5); ax.set_ylim(-0.05, 1.10)
        ax.set_title(panel_title(gid, sp['A_rho']))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(0, tmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.015,
                      pad=0.01)
    cb.set_label(r'$t/t_A$')
    savefig(fig, 'F17_coherence.pdf')


def fig_cross_phase(all_data, spectra, report=None):
    """F18: shell phase phi(k_perp) + signed r(k_perp) from spectra, 2x4."""
    if spectra is None:
        print('  [skip F18] spectra not reduced; rerun reduce with --spectra')
        return
    fig, axes = _make_2x4()
    tmax = max((sp['times'].max() for sp in spectra.values()), default=1.0)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        if gid not in spectra:
            ax.axis('off'); continue
        sp = spectra[gid]
        norm_t = Normalize(vmin=0, vmax=tmax); cmap_t = plt.cm.viridis
        axr = ax.twinx()
        axr.set_ylabel(r'$r(k_\perp)$'); axr.set_ylim(-1.05, 1.05)
        axr.axhline(0.0, color='0.45', lw=0.8, alpha=0.35)
        # sub-select ~3 epochs from the stored set for readability
        n_show = min(3, len(sp['times']))
        show = np.linspace(0, len(sp['times']) - 1, n_show, dtype=int)
        for j, i in enumerate(show):
            tv = sp['times'][i]
            ph = sp['phase'][i]; rr = sp['r'][i]
            mask = np.isfinite(ph) & np.isfinite(rr)
            if mask.sum() < 1:
                continue
            x = sp['k_over_kp0']
            ax.plot(x[mask], ph[mask], color=cmap_t(norm_t(tv)), lw=1.8,
                    alpha=0.92, label=(rf'$t={tv:.1f}$' if j == 0 else None))
            axr.plot(x[mask], rr[mask], color=cmap_t(norm_t(tv)), lw=1.2,
                     ls=':', alpha=0.85)
        for h, s in ((0.0, '-'), (np.pi / 2, '--'), (-np.pi / 2, '--')):
            ax.axhline(h, color='0.45', ls=s, lw=0.8, alpha=0.4)
        ax.axvline(1.0, color='k', ls='--', lw=1.0, alpha=0.55)
        ax.axvline(sp['kdiss_over_kp0'], color='r', ls='--', lw=1.0,
                   alpha=0.55)
        ax.set_xscale('log'); ax.set_xlabel(r'$k_\perp/k_{\perp,0}$')
        ax.set_ylabel(r'$\phi(k_\perp)$ [rad]')
        ax.set_xlim(left=0.5); ax.set_ylim(-np.pi, np.pi)
        ax.set_title(panel_title(gid, sp['A_rho']))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(0, tmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.015,
                      pad=0.01)
    cb.set_label(r'$t/t_A$')
    savefig(fig, 'F18_phase.pdf')


# Figure registry: name -> (function, needs_spectra) -----------------------------
FIG_REGISTRY = {
    'F01':  (fig_zplus_overlay,          False),
    'F02':  (fig_rho_waterfalls,         False),
    'F02b': (fig_reference_waterfall,    False),
    'F02c': (fig_rho_waterfalls_full,    False),
    'F03':  (fig_lperp_comparison,       False),
    'F04':  (fig_alpha_dynamic,          False),
    'F05':  (fig_alpha_inst,             False),
    'F06':  (fig_chi_A,                  False),
    'F09':  (fig_flux_collapse,          False),
    'F09b': (fig_flux_regression_clouds, False),
    'F10':  (fig_alpha_vs_chi,           False),
    'F10b': (fig_morphology_diagnostics, False),
    'F10c': (fig_gradient_thickness,     False),
    'F11':  (fig_erosion_raw,            False),
    'F11b': (fig_theory_vs_measurement,  False),
    'F12':  (fig_integrated_flux,        False),
    'F14':  (fig_slaved_closure,         False),
    'F14b': (fig_slaved_closure_measured, False),
    'F15':  (fig_Kperp_vs_dissipation,   False),
    'F16':  (fig_slaved_ratio,           False),
    'F17':  (fig_cross_coherence,        True),
    'F18':  (fig_cross_phase,            True),
}


# =============================================================================
#  SECTION 11 — REPORT
# =============================================================================

def write_regime_summary(all_data, report):
    report += ['\n' + '=' * 70,
               'REGIME SUMMARY  alpha_flux vs Lorentzian prediction',
               '=' * 70]
    for name, gids in (('STRONG', STRONG_GIDS), ('WEAK', WEAK_GIDS),
                       ('REFERENCE', REFERENCE_GIDS)):
        report.append(f'\n  --- {name} ---')
        for gid in gids:
            for tag, d in sorted(all_data.items()):
                if d['meta']['gid'] != gid or d['meta']['A_rho'] == 0:
                    continue
                fit = regress_alpha_from_flux(d)
                af = fit['alpha_fit']; cm = chi_A_mean(d)
                ap = float(lorentzian_alpha(cm))
                re = (af - ap) / ap if ap > 0 and np.isfinite(af) else np.nan
                report.append(
                    f'  {tag:24s}  chi_A={cm:6.2f}  '
                    f'a_flux={af:.4f}+/-{fit["alpha_err"]:.4f}  '
                    f'a_pred={ap:.4f}  rel_err={re:+.1%}  '
                    f'[{fit["status"]}]')


def write_report(report):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, REPORT_TXT)
    with open(path, 'w') as f:
        f.write('\n'.join(report))
    print(f'  Report -> {path}')


def _report_header():
    return [
        'Unified Density Transport Diagnostics — v13 MASTER',
        '=' * 70,
        f'Strong gids {STRONG_GIDS} (S1..S4) -> top row of 2x4 grids',
        f'Weak   gids {WEAK_GIDS} (W1..W4) -> bottom row of 2x4 grids',
        f'Reference gid {REFERENCE_GIDS} -> overlay-only + F02b',
        'Primary alpha source : regress_alpha_from_flux (SNR mask, transient '
        'cut, |X|^2 weights, bootstrap errors)',
        'Theory curve         : alpha_eff = (1/8) chi_A^2/(1+chi_A^2)',
        f'chi_A definition     : {CHI_CONFIG}',
        f'X_DISS={X_DISS} (PLM), N_y={NY_GRID}',
        'Per-group beta       : ' + ', '.join(
            f'{short_id(g)}:{GROUP_META[g].beta:g}' for g in ACTIVE_GIDS_ALL),
        f'Output dir : {OUTPUT_DIR}',
        '=' * 70, '',
    ]


# =============================================================================
#  SECTION 12 — CLI
# =============================================================================

def stage_reduce(do_spectra=False):
    print('\n=== STAGE 1: REDUCE ===')
    all_data, cat, load_log = load_all_runs()
    if not all_data:
        print('ERROR: no data loaded.'); sys.exit(1)
    save_reduced(all_data)
    if do_spectra:
        print('\nReducing cross-spectra ...')
        spectra = reduce_spectra(cat, all_data)
        save_spectra(spectra)
    if load_log:
        print('\nLoad warnings:')
        for msg in load_log:
            print('  ' + msg)
    return all_data


def stage_table(all_data=None):
    print('\n=== BRIDGE TABLE ===')
    if all_data is None:
        all_data = load_reduced()
    rows = build_bridge_table(all_data)
    save_bridge_table_csv(rows)
    save_bridge_table_latex(rows)
    return rows


def stage_plot(figs=None, style='manuscript', all_data=None):
    print('\n=== STAGE 2: PLOT ===')
    setup_mpl(style)
    if all_data is None:
        all_data = load_reduced()
    spectra = load_spectra()
    report = _report_header()
    names = figs if figs else list(FIG_REGISTRY)
    for name in names:
        if name not in FIG_REGISTRY:
            print(f'  [skip] unknown figure {name}'); continue
        func, needs_spectra = FIG_REGISTRY[name]
        print(f'{name}  {func.__name__} ...')
        try:
            if needs_spectra:
                func(all_data, spectra, report=report)
            else:
                func(all_data, report=report)
        except Exception as exc:
            print(f'  [FAIL] {name}: {exc}')
            report.append(f'[FAIL] {name}: {exc}')
    write_regime_summary(all_data, report)
    write_report(report)
    print(f'\nAll outputs in: {OUTPUT_DIR}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    p.add_argument('stage', nargs='?', default='all',
                   choices=['reduce', 'plot', 'table', 'all'])
    p.add_argument('--spectra', action='store_true',
                   help='also reduce cross-spectra (needed for F17/F18)')
    p.add_argument('--figs', nargs='+', default=None,
                   help='subset of figures to render, e.g. --figs F09 F10')
    p.add_argument('--style', default='manuscript',
                   choices=['manuscript', 'poster'])
    args = p.parse_args()

    if args.stage == 'reduce':
        stage_reduce(do_spectra=args.spectra)
    elif args.stage == 'table':
        stage_table()
    elif args.stage == 'plot':
        stage_plot(figs=args.figs, style=args.style)
    else:                                              # all
        all_data = stage_reduce(do_spectra=args.spectra)
        stage_table(all_data)
        stage_plot(figs=args.figs, style=args.style, all_data=all_data)


if __name__ == '__main__':
    main()
