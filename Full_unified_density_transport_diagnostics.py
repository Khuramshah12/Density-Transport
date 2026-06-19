"""
unified_density_transport_diagnostics_v12.py
=============================================
Regime-organised RMHD density-transport diagnostic suite.

------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES  (function-by-function map)
------------------------------------------------------------------------
CONFIG / GEOMETRY
  k_par_phys_of / k_perp_phys_of / l_par_phys_of : physical input
        wavenumbers and parallel integral scale from box geometry.
  k_diss_of / t_diss_inst : PPM dissipation wavenumber (X=40) and the
        instantaneous phase-mixing time to reach it.
  regime_of / is_strong / is_weak / is_reference / is_square :
        regime + profile-shape classifiers per group id (gid).

PER-SNAPSHOT REDUCTION
  process_snapshot : reads one .athdf file and returns, on the
        xz-averaged mean profile, the k=1/k=2 Fourier amplitudes
        (A_rho_k1, A_rho_k2), the direct mean-profile gradient
        d_y rho0 (for flux regression), the profile variance and
        k=2,3,4 harmonic powers (morphology), the Elsasser RMS
        amplitudes z+_rms / z-_rms and sigma_c, the perpendicular
        ACF correlation lengths (l_perp_y, l_perp_z, l_perp_full),
        the cross-field flux profile Gamma_rho(y) = <drho dv_y>_xz,
        and (drho/rho0)_rms with dB_par / dv_par RMS.
  fourier_mode_amplitude : harmonic-n amplitude of a 1D periodic profile.
  _acf_len_1d : Wiener-Khinchin integral autocorrelation scale.

DATA LOADING
  load_all_runs : assembles every run in OUTPUT_SPECS into time-series
        arrays + 2D (Nt,Ny) profile / flux / gradient arrays.

CLOSURE PHYSICS
  chi_A : turbulence-strength parameter chi_A(t)=z+ l_par/(v_A l_perp_y).
  alpha_theory_imbalance / alpha_theory_chiA : 1/[4(1+sigma_c)] -> 1/8
        baseline and the Lorentzian chi_A^2/(1+chi_A^2) bridge.
  build_prediction / build_prediction_chiA / build_prediction_chiA_weak :
        Fickian k=mode_n erosion predictors.
  Kperp_phase_mixing : phase-mixing wavenumber K_perp(t).
  integrate_abs_flux : int |Gamma_rho| dy.

PRIMARY TRANSPORT DIAGNOSTIC  
  _regressor_arrays : builds predictor X=-z+ l_perp,y d_y rho0 and
        response Y=Gamma_rho on the full (y,t) cloud.
  regress_alpha_from_flux : weighted no-intercept WLS slope alpha with
        SNR mask, transient cut, and bootstrap uncertainty.  THIS is the
        single source of every quoted alpha in the suite.
  regress_alpha_per_time : per-time-slice alpha_inst(t).
  alpha_flux : thin (alpha, rms) wrapper around regress_alpha_from_flux.
  coarse_erosion_alpha : apparent alpha from k=1 decay (consistency
        check only, NOT a transport-law calibration).

MORPHOLOGY
  compute_morphology : variance, k=1..4 amplitudes, gradient thickness
        delta_grad, and the squared-deviation spread.

BRIDGE TABLE
  compute_bridge_row / build_bridge_table / save_bridge_table_csv /
  save_bridge_table_latex : per-run <chi_A>, <sigma_c>, <eta_eff>,
        ACF anisotropy, <K_perp/k_diss>, and the flux-regression alpha.

FIGURES  (all per-group grids are 2x4: S1 S2 S3 S4 / W1 W2 W3 W4)
  F01  z+(t) overlay              F02  waterfalls (2x2 representative)
  F02b reference waterfall        F02c waterfalls (2x4 full suite)
  F03  correlation lengths        F04  coarse k=1 alpha(t) (2x4)
  F06  chi_A(t) overlay           F09  flux-law COLLAPSE (single panel)
  F09b flux-regression clouds     F10  alpha_eff vs chi_A (single panel)
  F10b morphology (2x4)           F11  k=1 erosion check (2x4)
  F11b k=1/k=2 erosion (2x4)      F12  integrated flux (2x4)
  F14  slaved closure (2x4)       F14b measured only (2x4)
  F15  K_perp vs k_diss (2x4)     F16  slaved ratio (2x4)
  F17  cross-coherence (2x4)      F18  cross-phase (2x4)

Physical conventions
--------------------------------
  z+_rms      sqrt(<z+_y^2 + z+_z^2>)     two-component vector RMS
  alpha_th    1/[4(1+sigma_c)] -> 1/8     prefactor at sigma_c=1
  l_perp_y    ACF_y(z+_y)                 for eta_turb
  v_A^2/c_s^2 2/(gamma*beta) = 1.2        for gamma=5/3, beta=1
  chi_A       z+_rms * l_par / (v_A * l_perp_y)
  X_DISS      40                          PPM dissipation calibration
"""

import os
import gc
import glob
import sys
import json
import csv
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

warnings.filterwarnings('ignore')

sys.path.insert(1, '/nesi/nobackup/uoo02637/khurram/density_transport/'
                   'Initial Conditions/')
import athena_read as aread


# =============================================================================
#  CONFIGURATION
# =============================================================================

BASE = ('/home/abbkh891/00_nesi_projects/uoo02637_nobackup/khurram/'
        'density_transport/500_k15/')

# -----------------------------------------------------------------------------
#  OUTPUT_SPECS — (A_rho, output_folder, gid). 
# -----------------------------------------------------------------------------

OUTPUT_SPECS = [
    # --- STRONG REGIME (sinusoidal) ------------------------------------------
    (0.00,  BASE + 'rho0_Lx5/output/',                  0),
    (0.025, BASE + 'rho0_025_Lx5/output/',              0),
    (0.05,  BASE + 'rho0_05_Lx5/output/',               0),
    (0.025, BASE + 'A02k110/rho0_025/output/',          1),
    (0.05,  BASE + 'A02k110/rho0_05/output/',           1),
    (0.05,  BASE + 'rho0_05_Lx10/output/',              2),
    (0.10,  BASE + 'rho0_1_Lx10/output/',               2),
    # --- STRONG SQUARE-WAVE (S4) --------------------------------------------
    (0.5,   BASE + 'Sharp_grad/rho0_5_kp15B01A02/output/', 3),

    # --- WEAK REGIME (sinusoidal) -------------------------------------------
    (0.00,  BASE + 'rho0/output/',                      4),
    (0.025, BASE + 'rho0_025/output/',                  4),
    (0.05,  BASE + 'rho0_05/output/',                   4),
    (0.05,  BASE + 'rho0_05A02kp2/output/',             5),
    (0.10,  BASE + 'rho0_1A02kp2/output/',              5),
    (0.025, BASE + 'test_Lx5kp3z0075/output/',          6),
    # --- WEAK SQUARE-WAVE (W4) ----------------------------------------------
    (0.5,   BASE + 'Sharp_grad/cubic_rho0_5kp15B01A02/output/', 7),

    # --- REFERENCE (chi_A ~ 1) ----------------------------------------------
    (0.25,  BASE + 'cubic_test_z03rho025kp7/output/',   8),
]

# -----------------------------------------------------------------------------
#  GROUP_META — one entry per gid.
# -----------------------------------------------------------------------------

GROUP_META = {
    # --- STRONG -------------------------------------------------------------
    0: dict(regime='strong', square=False,
            kappa_par=1, kappa_perp=5,  z0p=0.2,  Lx=5.0,  Ly=1.0, Lz=1.0,
            ls='-',               lw=2.2,
            label=r'S1: $L_x{=}5,\;\kappa_\perp{=}5$'),
    1: dict(regime='strong', square=False,
            kappa_par=1, kappa_perp=10, z0p=0.2,  Lx=5.0,  Ly=1.0, Lz=1.0,
            ls=(0, (3, 1, 1, 1)), lw=2.0,
            label=r'S2: $L_x{=}5,\;\kappa_\perp{=}10$'),
    2: dict(regime='strong', square=False,
            kappa_par=1, kappa_perp=5,  z0p=0.2,  Lx=10.0, Ly=1.0, Lz=1.0,
            ls=(0, (3, 5, 1, 5)), lw=2.2,
            label=r'S3: $L_x{=}10,\;\kappa_\perp{=}5$'),
    3: dict(regime='strong', square=True,
            kappa_par=1, kappa_perp=5,  z0p=0.2,  Lx=5.0,  Ly=1.0, Lz=1.0,
            ls=(0, (5, 2)),       lw=2.2,
            label=r'S4: square, $L_x{=}5,\;\kappa_\perp{=}5$'),

    # --- WEAK ---------------------------------------------------------------
    4: dict(regime='weak', square=False,
            kappa_par=1, kappa_perp=5,  z0p=0.2,  Lx=1.0,  Ly=1.0, Lz=1.0,
            ls='-',               lw=2.2,
            label=r'W1: $L_x{=}1,\;\kappa_\perp{=}5$'),
    5: dict(regime='weak', square=False,
            kappa_par=2, kappa_perp=5,  z0p=0.2,  Lx=1.0,  Ly=1.0, Lz=1.0,
            ls='--',              lw=2.2,
            label=r'W2: $L_x{=}1,\;\kappa_{\parallel}{=}2,\;\kappa_\perp{=}5$'),
    6: dict(regime='weak', square=False,
            kappa_par=1, kappa_perp=3,  z0p=0.075, Lx=5.0, Ly=1.0, Lz=1.0,
            ls=':',               lw=2.0,
            label=r'W3: $L_x{=}5,\;\kappa_\perp{=}3,\;z_0^+{=}0.075$'),
    7: dict(regime='weak', square=True,
            kappa_par=1, kappa_perp=5,  z0p=0.2,  Lx=1.0,  Ly=1.0, Lz=1.0,
            ls=(0, (5, 2)),       lw=2.0,
            label=r'W4: square, $L_x{=}1,\;\kappa_\perp{=}5$'),

    # --- REFERENCE ----------------------------------------------------------
    8: dict(regime='reference', square=False,
            kappa_par=1, kappa_perp=7,  z0p=0.3,  Lx=1.0,  Ly=1.0, Lz=1.0,
            ls='-.',              lw=2.4,
            label=r'R: $L_x{=}1,\;\kappa_\perp{=}7,\;z_0^+{=}0.3$'),
}

STRONG_GIDS    = [g for g, m in GROUP_META.items() if m['regime'] == 'strong']
WEAK_GIDS      = [g for g, m in GROUP_META.items() if m['regime'] == 'weak']
REFERENCE_GIDS = [g for g, m in GROUP_META.items() if m['regime'] == 'reference']

# 2x4 panel order: strong on top row, weak on bottom row.
ACTIVE_GIDS_2x4 = STRONG_GIDS + WEAK_GIDS          # exactly 8 entries
ACTIVE_GIDS     = ACTIVE_GIDS_2x4                  # alias used by panel loops
ACTIVE_GIDS_ALL = STRONG_GIDS + WEAK_GIDS + REFERENCE_GIDS

assert len(ACTIVE_GIDS_2x4) == 8, \
    f'2x4 grid requires exactly 8 active gids, got {len(ACTIVE_GIDS_2x4)}'

# Numerical / plot constants
X_DISS       = 40
NY_GRID      = 500
FILE_PATTERN = 'from_array.out.*.athdf'
OUTPUT_DIR   = BASE + 'unified_diagnostics_results_v12_ChatGPT/'
T_MAX_PLOT   = 50.0

# Physical constants
GAMMA_AD     = 5.0 / 3.0
BETA_PLASMA  = 1.0
VA2_OVER_CS2 = 2.0 / (GAMMA_AD * BETA_PLASMA)

# -----------------------------------------------------------------------------
#  COLOUR / STYLE MAPS
# -----------------------------------------------------------------------------

# Scheme A — A_rho-keyed (panel figures that overlay several A_rho per group)
COLOR_ARHO = {0.000: '#1f77b4', 0.025: '#e377c2',
              0.050: '#2ca02c', 0.100: '#d62728',
              0.250: '#b8860b', 0.500: '#9467bd'}

# Scheme B — gid-keyed (overlay-only figures F01 / F06); covers all 9 gids
GID_COLOR = {0: '#d62728', 1: '#ff7f0e', 2: '#8c564b', 3: '#e377c2',
             4: '#1f77b4', 5: '#17becf', 6: '#9467bd', 7: '#2ca02c',
             8: '#7f7f7f'}
GID_LS = {0: '-', 1: '--', 2: '-.', 3: (0, (5, 2)),
          4: ':', 5: (0, (3, 1, 1, 1)), 6: (0, (1, 1, 5, 1)),
          7: (0, (5, 1, 1, 1)), 8: (0, (4, 2, 1, 2))}
GID_LW = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.2, 4: 2.0, 5: 2.0, 6: 2.0,
          7: 2.2, 8: 2.6}

# Scheme C — regime/profile markers for the collapse + alpha-chi panels.
REGIME_COLOR = {'strong': '#c0392b', 'weak': '#2471a3',
                'reference': '#444444'}

REGIME_BADGE = {
    'strong':    dict(text='STRONG', bg='#ffe6e6', fg='#a30000'),
    'weak':      dict(text='WEAK',   bg='#e6f0ff', fg='#003a99'),
    'reference': dict(text='REF',    bg='#ececec', fg='#333333'),
}


# =============================================================================
#  MATPLOTLIB SETUP  (publication typography; minimums enforced)
#    axis labels >=16pt, ticks >=13pt, legend 13pt, titles >=15pt.
# =============================================================================

def setup_mpl():
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Computer Modern Roman', 'Times New Roman',
                                'DejaVu Serif'],
        'mathtext.fontset':    'cm',
        'font.size':           15,
        'axes.labelsize':      17,   # >= 14 pt requirement
        'axes.titlesize':      16,   # >= 14 pt requirement
        'legend.fontsize':     13,   # 12-14 pt requirement
        'legend.framealpha':   0.92,
        'legend.edgecolor':    '0.45',
        'xtick.labelsize':     13,   # >= 12 pt requirement
        'ytick.labelsize':     13,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.top':           True,
        'ytick.right':         True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.size':    5.5,
        'xtick.minor.size':    3.2,
        'ytick.major.size':    5.5,
        'ytick.minor.size':    3.2,
        'axes.linewidth':      1.2,
        'lines.linewidth':     2.2,
        'grid.alpha':          0.25,
        'grid.linewidth':      0.7,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.08,
        'figure.dpi':          120,
    })
    try:
        import subprocess
        subprocess.run(['latex', '--version'], capture_output=True, check=True)
        mpl.rcParams['text.usetex'] = True
    except Exception:
        mpl.rcParams['text.usetex'] = False


def setup_mpl_poster():
    """Poster-quality override (used only by F11b)."""
    mpl.rcParams.update({
        'font.size': 22, 'axes.labelsize': 26, 'axes.titlesize': 22,
        'axes.titleweight': 'bold', 'legend.fontsize': 16,
        'xtick.labelsize': 20, 'ytick.labelsize': 20,
        'axes.linewidth': 2.0, 'lines.linewidth': 3.0,
        'xtick.major.size': 8.0, 'ytick.major.size': 8.0,
        'xtick.major.width': 1.8, 'ytick.major.width': 1.8,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    })


def restore_mpl_default():
    setup_mpl()


# =============================================================================
#  REGIME / PROFILE HELPERS
# =============================================================================

def regime_of(gid):    return GROUP_META[gid]['regime']
def is_strong(gid):    return regime_of(gid) == 'strong'
def is_weak(gid):      return regime_of(gid) == 'weak'
def is_reference(gid): return regime_of(gid) == 'reference'
def is_square(gid):    return bool(GROUP_META[gid].get('square', False))

def short_id(gid):
    """Compact run-family identifier, e.g. 'S1', 'W4'."""
    return GROUP_META[gid]['label'].split(':')[0].strip()

def panel_title(gid, A_rho=None):
    """Subplot identifier title, e.g. S1_A0.05 (monospace, always visible)."""
    sid = short_id(gid)
    if A_rho is None:
        return rf'$\mathtt{{{sid}}}$'
    return rf'$\mathtt{{{sid}\_A{A_rho:g}}}$'


# =============================================================================
#  PHYSICAL WAVENUMBERS / TIMESCALES
# =============================================================================

def k_par_phys_of(gid):
    gm = GROUP_META[gid]
    return gm['kappa_par'] * 2.0 * np.pi / gm['Lx']

def k_perp_phys_of(gid):
    gm = GROUP_META[gid]
    return gm['kappa_perp'] * 2.0 * np.pi / gm['Ly']

def l_par_phys_of(gid):
    return 1.0 / k_par_phys_of(gid)

def k_diss_of(gid, X=X_DISS):
    gm = GROUP_META[gid]
    return 2.0 * np.pi * NY_GRID / (X * gm['Ly'])

def t_diss_inst(gid, dvAdy_max, X=X_DISS):
    if dvAdy_max < 1e-12:
        return np.inf
    return k_diss_of(gid, X) / (k_par_phys_of(gid) * dvAdy_max)


# =============================================================================
#  SPECTRAL HELPERS
# =============================================================================

def _acf_len_1d(field, axis, L):
    """Wiener-Khinchin integral autocorrelation scale along one axis."""
    N    = field.shape[axis]
    lags = np.arange(N) * (L / N)
    f    = field - np.mean(field, axis=axis, keepdims=True)
    Pk   = np.mean(np.abs(np.fft.fft(f, axis=axis)) ** 2,
                   axis=tuple(i for i in range(field.ndim) if i != axis)) / N ** 2
    ac = np.real(np.fft.ifft(Pk))
    if ac[0] <= 0:
        return np.nan
    R   = ac / ac[0]
    neg = np.where(R[1:N // 2] <= 0)[0]
    zc  = int(neg[0]) + 1 if len(neg) else N // 2
    try:
        return max(float(np.trapezoid(R[:zc + 1], lags[:zc + 1])), 0.0)
    except AttributeError:
        return max(float(np.trapz(R[:zc + 1], lags[:zc + 1])), 0.0)


def fourier_mode_amplitude(profile, y_arr, Ly, n):
    """Amplitude of harmonic n for a 1D periodic profile (sqrt(a_c^2+a_s^2))."""
    theta = 2.0 * np.pi * n * y_arr / Ly
    c = np.cos(theta);  s = np.sin(theta)
    try:
        a_c = (2.0 / Ly) * float(np.trapezoid(profile * c, y_arr))
        a_s = (2.0 / Ly) * float(np.trapezoid(profile * s, y_arr))
    except AttributeError:
        a_c = (2.0 / Ly) * float(np.trapz(profile * c, y_arr))
        a_s = (2.0 / Ly) * float(np.trapz(profile * s, y_arr))
    return float(np.sqrt(a_c * a_c + a_s * a_s))


# =============================================================================
#  SNAPSHOT PROCESSOR  (v10: k=1/k=2 amplitudes, d_y rho0, morphology)
# =============================================================================


def process_snapshot(fn, gid):
    """Scalar + profile diagnostics for one .athdf snapshot."""
    d = aread.athdf(fn)
    t = float(d['Time'])
    gm = GROUP_META[gid]

    rho = np.asarray(d['rho'], dtype=np.float64)
    vy = np.asarray(d['vel2'], dtype=np.float64)
    vz = np.asarray(d['vel3'], dtype=np.float64)
    By = np.asarray(d['Bcc2'], dtype=np.float64)
    Bz = np.asarray(d['Bcc3'], dtype=np.float64)
    vx = np.asarray(d['vel1'], dtype=np.float64)
    Bx = np.asarray(d['Bcc1'], dtype=np.float64)

    xz = (0, 2)
    try:
        ymin = float(d['RootGridX2'][0])
        ymax = float(d['RootGridX2'][1])
    except Exception:
        ymin, ymax = 0.0, gm['Ly']
    Ly = ymax - ymin
    y_arr = ymin + (np.arange(rho.shape[1]) + 0.5) * (Ly / rho.shape[1])
    try:
        Lz = float(d['RootGridX3'][1]) - float(d['RootGridX3'][0])
    except Exception:
        Lz = gm['Lz']

    rho0_3d = np.mean(rho, axis=xz, keepdims=True)
    rho0_1d = rho0_3d[0, :, 0]
    rho_mean = float(np.mean(rho0_1d))
    rho_norm = rho0_1d / (rho_mean + 1e-30) - 1.0

    vA_y = 1.0 / np.sqrt(rho0_1d + 1e-30)
    dvAdy_max = float(np.max(np.abs(np.gradient(vA_y, y_arr))))

    grad_log_rho0 = np.gradient(np.log(rho0_1d + 1e-30), y_arr)
    grad_rho0 = np.gradient(rho0_1d, y_arr, edge_order=2)
    Frho_rms_y = float(np.sqrt(np.mean(grad_log_rho0 ** 2)))
    Frho_max_y = float(np.max(np.abs(grad_log_rho0)))

    profile_var_norm = float(np.var(rho_norm))
    rho_power_k2 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 2) ** 2)
    rho_power_k3 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 3) ** 2)
    rho_power_k4 = float(fourier_mode_amplitude(rho_norm, y_arr, Ly, 4) ** 2)

    inv_sqr = 1.0 / np.sqrt(rho0_3d + 1e-30)
    delta_vy = vy - np.mean(vy, axis=xz, keepdims=True)
    delta_vz = vz - np.mean(vz, axis=xz, keepdims=True)
    delta_vx = vx - np.mean(vx, axis=xz, keepdims=True)
    delta_By = By - np.mean(By, axis=xz, keepdims=True)
    delta_Bz = Bz - np.mean(Bz, axis=xz, keepdims=True)
    delta_Bx = Bx - np.mean(Bx, axis=xz, keepdims=True)
    delta_rho = rho - rho0_3d

    zp_y = delta_vy + delta_By * inv_sqr
    zp_z = delta_vz + delta_Bz * inv_sqr
    zm_y = delta_vy - delta_By * inv_sqr
    zm_z = delta_vz - delta_Bz * inv_sqr

    zplus_rms = float(np.sqrt(np.mean(zp_y**2 + zp_z**2)))
    zminus_rms = float(np.sqrt(np.mean(zm_y**2 + zm_z**2)))
    E_p = zplus_rms**2
    E_m = zminus_rms**2
    sigma_c = (E_p - E_m) / (E_p + E_m) if (E_p + E_m) > 1e-30 else 0.0

    dvpar_rms = float(np.sqrt(np.mean(delta_vx**2)))
    dBpar_rms = float(np.sqrt(np.mean(delta_Bx**2)))

    drho_over_rho0_arr = delta_rho / (rho0_3d + 1e-30)
    drho_over_rho0_rms = float(np.sqrt(np.mean(drho_over_rho0_arr**2)))

    # real-space correlation coefficient r between density and transverse velocity
    num = float(np.mean(delta_rho * delta_vy))
    den = float(np.sqrt(np.mean(delta_rho**2) * np.mean(delta_vy**2)) + 1e-60)
    rho_vy_corr = num / den

    Gamma_rho_1d = np.mean(delta_rho * delta_vy, axis=xz)

    A_rho_k1 = fourier_mode_amplitude(rho_norm, y_arr, Ly, 1)
    A_rho_k2 = fourier_mode_amplitude(rho_norm, y_arr, Ly, 2)

    acf_y_zpy = _acf_len_1d(zp_y, axis=1, L=Ly)
    acf_y_zpz = _acf_len_1d(zp_z, axis=1, L=Ly)
    acf_z_zpy = _acf_len_1d(zp_y, axis=0, L=Lz)
    acf_z_zpz = _acf_len_1d(zp_z, axis=0, L=Lz)

    def _avg(a, b):
        if np.isfinite(a) and np.isfinite(b):
            return 0.5 * (a + b)
        return a if np.isfinite(a) else b

    l_perp_y_full = _avg(acf_y_zpy, acf_y_zpz)
    l_perp_z_full = _avg(acf_z_zpy, acf_z_zpz)
    l_perp_full = _avg(l_perp_y_full, l_perp_z_full)
    l_perp_y = acf_y_zpy
    l_perp_z = acf_z_zpy

    del rho, vy, vz, By, Bz, vx, Bx, delta_rho, delta_vy, delta_vz
    del delta_vx, delta_By, delta_Bz, delta_Bx, zp_y, zp_z, zm_y, zm_z
    del inv_sqr, drho_over_rho0_arr
    gc.collect()

    return dict(
        t=t, Ly=Ly, Lz=Lz, y_arr=y_arr,
        rho0=rho0_1d, rho_mean=rho_mean, rho_norm=rho_norm,
        drho0_dy=grad_rho0,
        profile_var_norm=profile_var_norm,
        rho_power_k2=rho_power_k2, rho_power_k3=rho_power_k3,
        rho_power_k4=rho_power_k4,
        dvAdy_max=dvAdy_max,
        Frho_rms_y=Frho_rms_y, Frho_max_y=Frho_max_y,
        zplus_rms=zplus_rms, zminus_rms=zminus_rms, sigma_c=sigma_c,
        dvpar_rms=dvpar_rms, dBpar_rms=dBpar_rms,
        drho_over_rho0_rms=drho_over_rho0_rms,
        rho_vy_corr=rho_vy_corr,
        Gamma_rho=Gamma_rho_1d,
        A_rho_k1=A_rho_k1, A_rho_k2=A_rho_k2,
        l_perp_y=l_perp_y, l_perp_z=l_perp_z,
        l_perp_y_full=l_perp_y_full, l_perp_z_full=l_perp_z_full,
        l_perp_full=l_perp_full,
    )


def load_all_runs():
    all_data = {}
    cat = {}

    for A_rho, folder, gid in OUTPUT_SPECS:
        if gid not in GROUP_META:
            print(f'  [SKIP] gid={gid} not in GROUP_META')
            continue
        tag = f'g{gid}_Arho{A_rho:.3f}'
        if not os.path.isdir(folder):
            print(f'  [SKIP] {tag}: folder not found ({folder})')
            continue
        snaps = sorted(glob.glob(os.path.join(folder, FILE_PATTERN)))
        if not snaps:
            print(f'  [SKIP] {tag}: no .athdf files in {folder}')
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
                print(f'  WARN: skipped {os.path.basename(fn)}: {exc}')

        if not records:
            continue
        records.sort(key=lambda r: r['t'])

        def _arr(key):
            return np.array([r[key] for r in records])

        t_arr = _arr('t')
        A_rho_k1 = _arr('A_rho_k1')
        A_rho_k2 = _arr('A_rho_k2')
        A0 = A_rho_k1[0] if A_rho_k1[0] > 1e-12 else 1.0
        gm = GROUP_META[gid]

        all_data[tag] = dict(
            meta=dict(
                A_rho=A_rho, gid=gid, regime=gm['regime'],
                square=gm.get('square', False),
                kappa_par=gm['kappa_par'], kappa_perp=gm['kappa_perp'],
                k_par_phys=k_par_phys_of(gid), k_perp_phys=k_perp_phys_of(gid),
                l_par_phys=l_par_phys_of(gid), k_diss=k_diss_of(gid),
                z0p=gm['z0p'], Lx=gm['Lx'], Ly=records[0]['Ly'],
                ls=gm['ls'], lw=gm['lw'], group_label=gm['label'],
            ),
            t=t_arr,
            A_rho=A_rho_k1, A_rho_norm=A_rho_k1 / A0,
            A_rho_k2=A_rho_k2, A_rho_k2_norm=A_rho_k2 / A0,
            zplus_rms=_arr('zplus_rms'), zminus_rms=_arr('zminus_rms'),
            sigma_c=_arr('sigma_c'),
            l_perp_y=_arr('l_perp_y'), l_perp_z=_arr('l_perp_z'),
            l_perp_y_full=_arr('l_perp_y_full'),
            l_perp_z_full=_arr('l_perp_z_full'), l_perp_full=_arr('l_perp_full'),
            dvAdy_max=_arr('dvAdy_max'),
            Frho_rms_y=_arr('Frho_rms_y'), Frho_max_y=_arr('Frho_max_y'),
            dvpar_rms=_arr('dvpar_rms'), dBpar_rms=_arr('dBpar_rms'),
            drho_over_rho0_rms=_arr('drho_over_rho0_rms'),
            rho_vy_corr=_arr('rho_vy_corr'),
            profile_var_norm=_arr('profile_var_norm'),
            rho_power_k2=_arr('rho_power_k2'), rho_power_k3=_arr('rho_power_k3'),
            rho_power_k4=_arr('rho_power_k4'),
            drho0_dy=np.vstack([r['drho0_dy'] for r in records]),
            rho0=np.vstack([r['rho0'] for r in records]),
            Gamma_rho=np.vstack([r['Gamma_rho'] for r in records]),
            y_arr=records[0]['y_arr'], Ly=records[0]['Ly'], A0=A0,
        )
        cat[tag] = dict(snapshots=snaps, gid=gid, A_rho=A_rho, folder=folder)
        del records; gc.collect()
    return all_data, cat

def chi_A(data):
    zp    = data['zplus_rms']
    lpy   = np.where(data['l_perp_y'] > 0, data['l_perp_y'], np.nan)
    l_par = data['meta']['l_par_phys']
    return zp * l_par / lpy

def alpha_theory_imbalance(sigma_c):
    return 1.0 / (4.0 * (1.0 + sigma_c))

def alpha_theory_chiA(sigma_c, chi):
    return alpha_theory_imbalance(sigma_c) * chi**2 / (1.0 + chi**2)

def build_prediction(data, alpha, mode_n=1):
    t = data['t']; zp = data['zplus_rms']; lpy = data['l_perp_y']
    Ky_n = mode_n * 2.0 * np.pi / data['Ly']
    lp = np.where(np.isfinite(lpy) & (lpy > 0), lpy, 0.0)
    eta = alpha * zp * lp
    I = np.zeros_like(t); I[1:] = cumtrapz(eta, t)
    return I, np.exp(-Ky_n**2 * I), eta

def build_prediction_chiA(data, alpha_th_const, mode_n=1):
    t = data['t']; zp = data['zplus_rms']; lpy = data['l_perp_y']
    Ky_n = mode_n * 2.0 * np.pi / data['Ly']
    chi = chi_A(data); cs = np.where(np.isfinite(chi), chi, 0.0)
    sup = cs**2 / (1.0 + cs**2)
    lp = np.where(np.isfinite(lpy) & (lpy > 0), lpy, 0.0)
    eta = alpha_th_const * sup * zp * lp
    I = np.zeros_like(t); I[1:] = cumtrapz(eta, t)
    return I, np.exp(-Ky_n**2 * I), eta

def build_prediction_chiA_weak(data, alpha_th_const, mode_n=1):
    t = data['t']; zp = data['zplus_rms']; lpy = data['l_perp_y']
    Ky_n = mode_n * 2.0 * np.pi / data['Ly']
    chi = chi_A(data); cs = np.where(np.isfinite(chi), chi, 0.0)
    sup = cs**2
    lp = np.where(np.isfinite(lpy) & (lpy > 0), lpy, 0.0)
    eta = alpha_th_const * sup * zp * lp
    I = np.zeros_like(t); I[1:] = cumtrapz(eta, t)
    return I, np.exp(-Ky_n**2 * I), eta

def Kperp_phase_mixing(data):
    gm = data['meta']; t = data['t']
    rate = gm['k_par_phys'] * data['dvAdy_max']
    integ = np.zeros_like(t); integ[1:] = cumtrapz(rate, t)
    return gm['k_perp_phys'] + integ

def integrate_abs_flux(data):
    Gamma = data['Gamma_rho']; y = data['y_arr']
    try:
        return np.array([float(np.trapezoid(np.abs(g), y)) for g in Gamma])
    except AttributeError:
        return np.array([float(np.trapz(np.abs(g), y)) for g in Gamma])


# =============================================================================
#  PRIMARY TRANSPORT DIAGNOSTIC  (v9 engine, verbatim behaviour)
#    Gamma_rho(y,t) = -alpha * z+_rms(t) * l_perp,y(t) * d_y rho0(y,t)
# =============================================================================

def _regressor_arrays(data):
    """Predictor X = -z+ l_perp,y d_y rho0 and response Y = Gamma_rho,
    both shaped (Nt, Ny)."""
    y        = data['y_arr']
    rho0_yt  = data['rho0']
    Gamma_yt = data['Gamma_rho']
    zp       = data['zplus_rms']
    lpy      = data['l_perp_y']
    drho_dy  = np.gradient(rho0_yt, y, axis=1)
    X = -zp[:, None] * lpy[:, None] * drho_dy
    Y = Gamma_yt
    valid_t = np.isfinite(lpy) & (lpy > 0)
    valid   = np.broadcast_to(valid_t[:, None], X.shape).copy()
    return X, Y, valid


def regress_alpha_from_flux(data, t_skip_frac=0.05, snr_frac=0.05,
                            weight_power=2.0, n_bootstrap=200, rng_seed=42):
    """Weighted no-intercept WLS slope alpha of Y=alpha*X over the (y,t)
    cloud, with |X|^weight_power weighting, SNR mask, transient cut, and a
    bootstrap 1-sigma uncertainty.  Returns a dict including X, Y, mask for
    plotting."""
    X, Y, valid = _regressor_arrays(data)
    t = data['t']; Nt, Ny = X.shape
    t_skip = int(np.ceil(t_skip_frac * Nt))
    valid[:t_skip, :] = False

    X_abs_max = np.max(np.abs(X[valid])) if valid.any() else 0.0
    keep = valid & (np.abs(X) >= snr_frac * X_abs_max)

    n_total = int(valid.sum()); n_used = int(keep.sum())
    t_window = (float(t[min(t_skip, Nt-1)]), float(t[-1]))

    if n_used < 10:
        return dict(alpha_fit=np.nan, alpha_err=np.nan, alpha_err_wls=np.nan,
                    residual_rms=np.nan, r_squared=np.nan, n_used=n_used,
                    n_total=n_total, t_window=t_window, X=X, Y=Y, mask=keep)

    Xk = X[keep]; Yk = Y[keep]; Wk = np.abs(Xk) ** weight_power
    Sxx = np.sum(Wk * Xk * Xk); Sxy = np.sum(Wk * Xk * Yk)
    alpha_fit = Sxy / Sxx
    resid = Yk - alpha_fit * Xk
    residual_rms = float(np.sqrt(np.mean(resid * resid)))
    sigma2_w = np.sum(Wk * resid * resid) / max(np.sum(Wk) - 1.0, 1.0)
    alpha_err_wls = float(np.sqrt(sigma2_w / Sxx))
    Yw = np.sum(Wk * Yk) / np.sum(Wk)
    SS_tot = np.sum(Wk * (Yk - Yw) ** 2); SS_res = np.sum(Wk * resid ** 2)
    r2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else np.nan

    rng = np.random.default_rng(rng_seed); n = len(Xk); boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = np.sum(Wk[idx]*Xk[idx]*Yk[idx]) / np.sum(Wk[idx]*Xk[idx]*Xk[idx])
    alpha_err = float(np.std(boot, ddof=1))

    return dict(alpha_fit=float(alpha_fit), alpha_err=alpha_err,
                alpha_err_wls=alpha_err_wls, residual_rms=residual_rms,
                r_squared=float(r2), n_used=n_used, n_total=n_total,
                t_window=t_window, X=X, Y=Y, mask=keep)


def regress_alpha_per_time(data, snr_frac=0.05, weight_power=2.0):
    """Per-time-slice alpha_inst(t) from the spatial (y) regression."""
    X, Y, valid = _regressor_arrays(data)
    Nt = X.shape[0]; alpha_t = np.full(Nt, np.nan)
    for i in range(Nt):
        if not valid[i].any():
            continue
        Xi, Yi = X[i], Y[i]; Xmax = np.max(np.abs(Xi))
        if Xmax <= 0:
            continue
        keep = np.abs(Xi) >= snr_frac * Xmax
        if keep.sum() < 4:
            continue
        Wi = np.abs(Xi[keep]) ** weight_power
        Sxx = np.sum(Wi * Xi[keep] ** 2); Sxy = np.sum(Wi * Xi[keep] * Yi[keep])
        if Sxx > 0:
            alpha_t[i] = Sxy / Sxx
    return alpha_t


def alpha_flux(data):
    """(alpha, rms) wrapper — the single source of every quoted alpha."""
    fit = regress_alpha_from_flux(data)
    return fit['alpha_fit'], fit['residual_rms']


def coarse_erosion_alpha(data):
    """Apparent alpha from the k=1 decay rate (consistency check only)."""
    t = data['t']; A = data['A_rho']; zp = data['zplus_rms']
    lpy = data['l_perp_y']; Ky = 2.0 * np.pi / data['Ly']
    logA = np.log(np.maximum(np.abs(A), 1e-16))
    dlogA = np.gradient(logA, t)
    denom = Ky**2 * zp * np.where(lpy > 0, lpy, np.nan)
    return -dlogA / denom

dynamic_alpha = coarse_erosion_alpha


# =============================================================================
#  MORPHOLOGY
# =============================================================================

def compute_morphology(data, kmodes=(1, 2, 3, 4)):
    """variance(t), k=1..4 amplitudes, gradient thickness delta_grad(t),
    and squared-deviation spread(t) of the mean profile."""
    t = data['t']; y = data['y_arr']; rho_yt = data['rho0']; Ly = data['Ly']
    rho_m = np.mean(rho_yt, axis=1)
    var_t = np.var(rho_yt, axis=1)
    A_modes = {}; norm = rho_yt / rho_m[:, None] - 1.0
    for k in kmodes:
        sin_k = np.sin(2.0*np.pi*k*y/Ly); cos_k = np.cos(2.0*np.pi*k*y/Ly)
        try:
            As = (2.0/Ly)*np.trapezoid(norm*sin_k[None, :], y, axis=1)
            Ac = (2.0/Ly)*np.trapezoid(norm*cos_k[None, :], y, axis=1)
        except AttributeError:
            As = (2.0/Ly)*np.trapz(norm*sin_k[None, :], y, axis=1)
            Ac = (2.0/Ly)*np.trapz(norm*cos_k[None, :], y, axis=1)
        A_modes[k] = np.sqrt(As**2 + Ac**2)
    drho_dy = np.gradient(rho_yt, y, axis=1)
    rng_t = rho_yt.max(axis=1) - rho_yt.min(axis=1)
    grad_t = np.max(np.abs(drho_dy), axis=1)
    delta_g = np.where(grad_t > 1e-14, rng_t / grad_t, np.nan)
    delta_rho = rho_yt - rho_m[:, None]; w = delta_rho**2
    W = np.sum(w, axis=1, keepdims=True); y_b = y[None, :]
    y_c = np.where(W > 0, np.sum(w*y_b, axis=1, keepdims=True)/W, 0.5*Ly)
    spread = np.where(W[:, 0] > 0,
                      np.sqrt(np.sum(w*(y_b - y_c)**2, axis=1)/W[:, 0]), np.nan)
    return dict(t=t, var=var_t, A_modes=A_modes, delta_grad=delta_g, spread=spread)


# =============================================================================
#  UTILITIES
# =============================================================================

def _color(A_rho):  return COLOR_ARHO.get(A_rho, '#888888')
def _tick_both(ax): ax.tick_params(which='both', direction='in', top=True, right=True)

def savefig(fig, name, dpi=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi) if dpi else fig.savefig(path)
    plt.close(fig); print(f'  -> {path}')

def _add_regime_badge(ax, gid, loc='upper left'):
    b = REGIME_BADGE[regime_of(gid)]
    pos = {'upper left': (0.025, 0.965, 'left', 'top'),
           'upper right': (0.975, 0.965, 'right', 'top'),
           'lower left': (0.025, 0.035, 'left', 'bottom'),
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
        if g not in present: continue
        if (not include_reference) and is_reference(g): continue
        gm = GROUP_META[g]
        s.append(Line2D([0], [0], color='k', lw=gm['lw'], ls=gm['ls'],
                        label=gm['label']))
    return c, s

def _legend_handles_by_gid(all_data, include_reference=True):
    present = sorted({d['meta']['gid'] for d in all_data.values()})
    h = []
    for g in present:
        if (not include_reference) and is_reference(g): continue
        h.append(Line2D([0], [0], color=GID_COLOR[g], lw=GID_LW[g],
                        ls=GID_LS[g], label=GROUP_META[g]['label']))
    return h

def _make_2x4(figsize=(19.5, 8.8), sharex=False, sharey=False):
    """2x4 grid: strong on TOP row, weak on BOTTOM row."""
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=sharex,
                             sharey=sharey, constrained_layout=True)
    return fig, axes.flatten()

def _safe_key(tag):
    return tag.replace('.', 'p').replace('-', 'n').replace(' ', '_')

# --- regime/profile marker scheme (collapse + alpha-chi panels) --------------

def marker_for(gid):
    """Scatter style: strong=circle, weak=square, square-wave=open,
    reference=star."""
    if is_reference(gid):
        return dict(marker='*', open=False, color=REGIME_COLOR['reference'], size=320)
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
                   edgecolors='k', linewidths=0.7, alpha=alpha, label=label,
                   zorder=zorder)

def _regime_profile_legend():
    return [
        Line2D([0], [0], marker='o', color=REGIME_COLOR['strong'], lw=0,
               markeredgecolor='k', markersize=12, label=r'Strong (sinusoidal)'),
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
    """Highest-A_rho gradient run of a group (for single-run panels)."""
    runs = [(tag, d) for tag, d in all_data.items()
            if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0]
    if not runs:
        return None, None
    return sorted(runs, key=lambda kv: kv[1]['meta']['A_rho'])[-1]


# =============================================================================
#  F01 — z+(t) overlay (gid-keyed; legend inside lower-left)
# =============================================================================

def fig_zplus_overlay(all_data):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    for tag, d in all_data.items():
        gid = d['meta']['gid']; t = d['t']; zp = d['zplus_rms']
        mk = t <= T_MAX_PLOT; tp = t[mk]; zpp = zp[mk]
        zp0 = zpp[0] if len(zpp) and zpp[0] > 1e-12 else 1.0
        ax.semilogy(tp, zpp / zp0, color=GID_COLOR[gid], ls=GID_LS[gid],
                    lw=GID_LW[gid], alpha=0.92)
        t_d = t_diss_inst(gid, d['dvAdy_max'][0])
        if np.isfinite(t_d) and 0 < t_d <= T_MAX_PLOT:
            ax.axvline(t_d, color=GID_COLOR[gid], ls=GID_LS[gid], lw=0.9, alpha=0.30)
    ax.set_xlim(0, T_MAX_PLOT)
    ax.set_xlabel(r'$t/t_A$')
    ax.set_ylabel(r'$z^+_{\rm rms}(t)/z^+_{\rm rms}(0)$')
    _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    ax.legend(handles=_legend_handles_by_gid(all_data), loc='lower left',
              framealpha=0.92, ncol=2)
    savefig(fig, 'F01_zplus_overlay.pdf')


# =============================================================================
#  F02 — waterfalls, representative 2x2: S1,S4 / W1,W4
# =============================================================================

def _waterfall_panel(ax, d, gid):
    t = d['t']; t_max = t.max() if t.size else 1.0
    norm = Normalize(vmin=0, vmax=t_max); cmap = plt.cm.viridis
    stride = max(1, len(t) // 45)
    for i in range(0, len(t), stride):
        ax.plot(d['y_arr'], d['rho0'][i], color=cmap(norm(t[i])), lw=0.9, alpha=0.70)
    ax.plot(d['y_arr'], d['rho0'][0], color='k', lw=2.8, zorder=6, label=r'$t=0$')
    ax.set_xlabel(r'$y$'); ax.set_ylabel(r'$\langle\rho\rangle_{xz}(y,t)$')
    ax.set_title(panel_title(gid, d['meta']['A_rho']))
    _tick_both(ax); ax.grid(True, alpha=0.20); ax.legend(loc='best')
    _add_regime_badge(ax, gid, loc='upper right')
    return norm, cmap

def fig_rho_waterfalls(all_data):
    order = [0, 3, 4, 7]  # S1, S4(sq), W1, W4(sq)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    axes = axes.flatten(); last = None
    for ax, gid in zip(axes, order):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        last = _waterfall_panel(ax, d, gid)
    if last:
        sm = ScalarMappable(cmap=last[1], norm=last[0]); sm.set_array([])
        cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.02, pad=0.01)
        cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02_rho_waterfalls_representative.pdf')


# =============================================================================
#  F02b — reference waterfall (standalone)
# =============================================================================

def fig_reference_waterfall(all_data):
    refs = [d for d in all_data.values()
            if is_reference(d['meta']['gid']) and d['meta']['A_rho'] > 0]
    if not refs:
        print('  [skip F02b] no reference runs'); return
    d = sorted(refs, key=lambda r: -r['meta']['A_rho'])[0]
    gid = d['meta']['gid']
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    norm, cmap = _waterfall_panel(ax, d, gid)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02b_reference_waterfall.pdf')


# =============================================================================
#  F02c — waterfalls, full suite 2x4: S1..S4 / W1..W4
# =============================================================================

def fig_rho_waterfalls_full(all_data):
    fig, axes = _make_2x4(figsize=(20.0, 9.5)); last = None
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        last = _waterfall_panel(ax, d, gid)
    if last:
        sm = ScalarMappable(cmap=last[1], norm=last[0]); sm.set_array([])
        cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.015, pad=0.01)
        cb.set_label(r'$t/t_A$')
    savefig(fig, 'F02c_rho_waterfalls_full_2x4.pdf')


# =============================================================================
#  F03 — correlation lengths (1x3 overlay)
# =============================================================================

def fig_lperp_comparison(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    keys = ['l_perp_y', 'l_perp_z', 'l_perp_full']
    titles = [r'$\ell_{\perp,y}$', r'$\ell_{\perp,z}$', r'$\ell_{\perp,\rm full}$']
    for ax, key, ttl in zip(axes, keys, titles):
        for tag, d in all_data.items():
            m = d['meta']
            ax.semilogy(d['t'], d[key], color=_color(m['A_rho']), ls=m['ls'],
                        lw=m['lw'], alpha=0.88)
        ax.set_xlabel(r'$t/t_A$'); ax.set_title(ttl)
        _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel(r'$\ell_\perp$')
    c_h, s_h = _legend_handles(all_data)
    axes[2].legend(handles=c_h + s_h, loc='lower left', fontsize=11, ncol=2)
    savefig(fig, 'F03_lperp_comparison.pdf')


# =============================================================================
#  F04 — coarse k=1 alpha(t) (2x4, consistency check)
# =============================================================================

def fig_alpha_dynamic(all_data, report_lines):
    fig, axes = _make_2x4(sharey=True)
    report_lines += ['\n' + '=' * 70,
                     'COARSE EROSION OBSERVABLE  (k=1 alpha consistency check)',
                     '=' * 70]
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']
            ax.plot(d['t'], coarse_erosion_alpha(d), color=_color(m['A_rho']),
                    lw=2.2, label=rf'$A_\rho={m["A_rho"]:.3g}$')
            af, rms = alpha_flux(d); chi_m = float(np.nanmean(chi_A(d)))
            report_lines.append(
                f'  {tag:24s}  [{regime_of(gid):>9s}]  alpha_flux={af:.4f}  '
                f'chi_A={chi_m:.2f}')
        ax.axhline(1/8, color='k', ls='--', lw=1.0, alpha=0.5, label=r'$1/8$')
        ax.set_title(panel_title(gid, A_repr))
        ax.set_xlim(left=0); ax.set_ylim(0, 0.42); ax.set_xlabel(r'$t/t_A$')
        _tick_both(ax); ax.grid(True, alpha=0.22); ax.legend(fontsize=11)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$\alpha_{k=1}(t)$'); axes[4].set_ylabel(r'$\alpha_{k=1}(t)$')
    savefig(fig, 'F04_coarse_alpha_k1.pdf')


# =============================================================================
#  F06 — chi_A(t) overlay (single panel; legend inside upper-right)
# =============================================================================

def fig_chi_A(all_data, report_lines):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    for tag, d in all_data.items():
        gid = d['meta']['gid']
        ax.semilogy(d['t'], chi_A(d), color=GID_COLOR[gid], ls=GID_LS[gid],
                    lw=GID_LW[gid], alpha=0.92)
    ax.axhspan(1e-3, 1.0, color=REGIME_BADGE['weak']['bg'], alpha=0.45, zorder=0)
    ax.axhspan(1.0, 1e2, color=REGIME_BADGE['strong']['bg'], alpha=0.45, zorder=0)
    ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel(r'$t/t_A$')
    ax.set_ylabel(r'$\chi_A=z^+_{\rm rms}\,\ell_{\parallel,0}/(v_A\,\ell_{\perp,y})$')
    ax.set_xlim(left=0); ax.set_ylim(3e-2, 1e2)
    _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
    handles = _legend_handles_by_gid(all_data) + [
        Patch(facecolor=REGIME_BADGE['weak']['bg'], alpha=0.55, label=r'$\chi_A<1$'),
        Patch(facecolor=REGIME_BADGE['strong']['bg'], alpha=0.55, label=r'$\chi_A>1$')]
    ax.legend(handles=handles, loc='upper right', framealpha=0.92, ncol=2)
    report_lines += ['\n' + '=' * 70, 'CHI_A — time-averaged', '=' * 70]
    for tag, d in sorted(all_data.items()):
        if d['meta']['A_rho'] == 0: continue
        cm = float(np.nanmean(chi_A(d)))
        report_lines.append(f'  {tag:24s}  <chi_A>={cm:5.2f}')
    savefig(fig, 'F06_chi_A.pdf')


# =============================================================================
#  F09 — direct-flux-law COLLAPSE (single large panel)
#    All runs on one (predictor, Gamma_rho) plane; markers encode
#    regime + profile.  Emphasises the transport law, not individual runs.
# =============================================================================

def fig_flux_collapse(all_data, report_lines):
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    report_lines += ['\n' + '=' * 70,
                     'DIRECT FLUX-LAW COLLAPSE  (single panel)', '=' * 70]
    rng = np.random.default_rng(7)
    all_X, all_Y = [], []
    for tag, d in sorted(all_data.items(), key=lambda kv: kv[1]['meta']['gid']):
        m = d['meta']
        if m['A_rho'] == 0: continue
        fit = regress_alpha_from_flux(d)
        X = fit['X'][fit['mask']].ravel(); Y = fit['Y'][fit['mask']].ravel()
        if X.size == 0: continue
        # subsample for a readable cloud
        if X.size > 400:
            idx = rng.choice(X.size, 400, replace=False); X, Y = X[idx], Y[idx]
        _scatter_run(ax, X, Y, m['gid'], size=22, alpha=0.30, zorder=3)
        all_X.append(fit['X'][fit['mask']].ravel())
        all_Y.append(fit['Y'][fit['mask']].ravel())
        report_lines.append(
            f'  {tag:24s}  [{regime_of(m["gid"]):>9s}]  '
            f'alpha={fit["alpha_fit"]:.4f} +/- {fit["alpha_err"]:.4f}  '
            f'R2={fit["r_squared"]:.3f}')
    # reference slope alpha_th = 1/8 across the global predictor range
    if all_X:
        Xc = np.concatenate(all_X)
        xs = np.linspace(np.nanpercentile(Xc, 1), np.nanpercentile(Xc, 99), 100)
        ax.plot(xs, (1/8.) * xs, 'k--', lw=2.2, zorder=6,
                label=r'$\alpha_{\rm th}=1/8$')
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


# =============================================================================
#  F09b — flux-regression clouds per group (2x4 hexbin; v9 primary engine)
# =============================================================================

def fig_flux_regression_clouds(all_data):
    fig, axes = _make_2x4(figsize=(20.0, 9.5))
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        fit = regress_alpha_from_flux(d)
        Xk = fit['X'][fit['mask']]; Yk = fit['Y'][fit['mask']]
        if Xk.size < 5:
            ax.axis('off'); continue
        ax.hexbin(Xk, Yk, gridsize=38, cmap='viridis', mincnt=1, linewidths=0)
        xs = np.linspace(Xk.min(), Xk.max(), 80)
        ax.plot(xs, fit['alpha_fit'] * xs, color='#d62728', lw=2.6,
                label=rf'$\alpha={fit["alpha_fit"]:.3f}\pm{fit["alpha_err"]:.3f}$')
        sm = float(np.nanmean(d['sigma_c'])); a_th = alpha_theory_imbalance(sm)
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


# =============================================================================
#  F10 — alpha_eff vs chi_A (single panel; colour=regime, marker=profile)
# =============================================================================

def fig_alpha_vs_chi(all_data, report_lines):
    fig, ax = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
    report_lines += ['\n' + '=' * 70,
                     'ALPHA_FLUX vs CHI_A  (Lorentzian theory)', '=' * 70]
    ax.axvspan(1e-2, 0.7, color=REGIME_BADGE['weak']['bg'], alpha=0.35, zorder=0)
    ax.axvspan(1.5, 1e2, color=REGIME_BADGE['strong']['bg'], alpha=0.35, zorder=0)
    cg = np.logspace(-1.5, 1.7, 200)
    ax.plot(cg, np.full_like(cg, 1/8.), 'k--', lw=1.4, alpha=0.6,
            label=r'$\alpha_{\rm th}=1/8$')
    ax.plot(cg, cg**2 / (8 * (1 + cg**2)), 'k-', lw=2.0, alpha=0.9,
            label=r'$\alpha_{\rm eff}=\tfrac{1}{8}\chi_A^2/(1+\chi_A^2)$')
    ax.axvline(1.0, color='k', ls=':', lw=0.9, alpha=0.5)
    for tag, d in all_data.items():
        m = d['meta']
        if m['A_rho'] == 0: continue
        af, _ = alpha_flux(d); cm = float(np.nanmean(chi_A(d)))
        if not np.isfinite(af): continue
        _scatter_run(ax, cm, af, m['gid'])
        a_pred = (1/8.) * cm**2 / (1 + cm**2)
        re = (af - a_pred)/a_pred if a_pred > 0 else np.nan
        report_lines.append(
            f'  {tag:24s}  [{regime_of(m["gid"]):>9s}]  chi_A={cm:5.2f}  '
            f'alpha={af:.4f}  pred={a_pred:.4f}  rel_err={re:+.1%}')
    ax.set_xscale('log'); ax.set_xlabel(r'$\langle\chi_A\rangle$')
    ax.set_ylabel(r'$\alpha_{\rm eff}$')
    ax.set_ylim(0, 0.30); ax.set_xlim(0.1, 30)
    _tick_both(ax); ax.grid(True, alpha=0.22)
    th = [Line2D([0], [0], color='k', ls='-', lw=2.0,
                 label=r'$\alpha_{\rm eff}(\chi_A)$'),
          Line2D([0], [0], color='k', ls='--', lw=1.4, label=r'$1/8$')]
    ax.legend(handles=_regime_profile_legend() + th, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=12)
    savefig(fig, 'F10_alpha_vs_chi.pdf')


# =============================================================================
#  F10b — morphology (2x4): variance + k=2,3,4 harmonics prominent;
#         gradient thickness DE-EMPHASISED (thin grey, low zorder).
# =============================================================================

def fig_morphology_diagnostics(all_data):
    fig, axes = _make_2x4(figsize=(20.0, 9.5), sharey=False)
    LS_K = {2: '--', 3: '-.', 4: ':'}
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        tag, d = _representative_run(all_data, gid)
        if d is None:
            ax.axis('off'); continue
        mo = compute_morphology(d); t = mo['t']
        A1_0 = mo['A_modes'][1][0] if mo['A_modes'][1][0] > 1e-14 else 1.0

        # variance (prominent, black)
        var_norm = mo['var'] / max(mo['var'][0], 1e-30)
        ax.semilogy(t, np.maximum(np.sqrt(var_norm), 1e-4), color='k', lw=2.8,
                    zorder=6, label=r'$\sqrt{\mathrm{Var}/\mathrm{Var}_0}$')
        # k=1 (prominent, colour)
        ax.semilogy(t, np.maximum(mo['A_modes'][1] / A1_0, 1e-4),
                    color=_color(d['meta']['A_rho']), lw=2.6, zorder=5,
                    label=r'$k{=}1$')
        # harmonics k=2,3,4 (medium, colour)
        for k in (2, 3, 4):
            ax.semilogy(t, np.maximum(mo['A_modes'][k] / A1_0, 1e-4),
                        color=_color(d['meta']['A_rho']), ls=LS_K[k], lw=1.8,
                        alpha=0.9, zorder=4, label=rf'$k{{=}}{k}$')
        # gradient thickness (DE-EMPHASISED: thin grey, faint, behind)
        dg = mo['delta_grad']; dg0 = dg[0] if np.isfinite(dg[0]) and dg[0] > 0 else 1.0
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


# =============================================================================
#  F10c — gradient thickness (2x4): square columns highlighted, others grey
# =============================================================================

def fig_gradient_thickness(all_data):
    fig, axes = _make_2x4(figsize=(20.0, 8.8))
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        emphasise = is_square(gid)
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']
            mo = compute_morphology(d)
            col = _color(m['A_rho']) if emphasise else '0.55'
            lw = 2.6 if emphasise else 1.4
            al = 0.95 if emphasise else 0.6
            ax.plot(mo['t'], mo['delta_grad'], color=col, lw=lw, alpha=al,
                    label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$')
        ax.set_ylabel(r'$\delta_{\rm grad}=\Delta\rho_0/\max|\partial_y\rho_0|$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        if emphasise:
            ax.legend(loc='best', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F10c_gradient_thickness.pdf')


# =============================================================================
#  F11 — k=1 erosion ratio (2x4, consistency check)
# =============================================================================

def fig_erosion_raw(all_data):
    fig, axes = _make_2x4(sharey=True)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']
            ax.semilogy(d['t'], d['A_rho_norm'], color=_color(m['A_rho']), lw=2.5,
                        label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.4)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylim(1e-3, 2.0); ax.set_xlim(left=0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='lower left', fontsize=11, framealpha=0.90)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$A_\rho(t)/A_\rho(0)$ (consistency check)')
    axes[4].set_ylabel(r'$A_\rho(t)/A_\rho(0)$ (consistency check)')
    savefig(fig, 'F11_k1_erosion_check.pdf')


# =============================================================================
#  F11b — k=1 / k=2 erosion vs flux-calibrated prediction (2x4, poster)
# =============================================================================

def compute_F11b_data(all_data, save=True):
    F11b = {}
    for tag, d in all_data.items():
        m = d['meta']
        if m['A_rho'] == 0: continue
        t = d['t']; A_k1 = d['A_rho_norm']; A_k2 = d['A_rho_k2_norm']
        sm = float(np.nanmean(d['sigma_c'])); a_th = alpha_theory_imbalance(sm)
        a_fx, _ = alpha_flux(d)          # flux-calibrated alpha (v9 engine)
        chi_m = float(np.nanmean(chi_A(d))); reg = regime_of(m['gid'])
        modes = {}
        for n in (1, 2):
            md = dict(A_meas=(A_k1 if n == 1 else A_k2))
            if reg == 'strong':
                _, md['P_const'], _ = build_prediction(d, a_th, mode_n=n)
                md['P_bf'] = (build_prediction(d, a_fx, mode_n=n)[1]
                              if np.isfinite(a_fx) else None)
                md['P_chi2'] = None
            else:
                _, md['P_chi2'], _ = build_prediction_chiA_weak(d, a_th, mode_n=n)
                md['P_const'] = None; md['P_bf'] = None
            modes[n] = md
        F11b[tag] = dict(gid=int(m['gid']), regime=reg, A_rho=float(m['A_rho']),
                         chi_A_mean=chi_m, alpha_th=a_th, alpha_bf=a_fx,
                         t=t, modes=modes)
    if save:
        _save_F11b_data(F11b)
    return F11b

def _save_F11b_data(F11b, npz='F11b_data.npz', js='F11b_data.json'):
    os.makedirs(OUTPUT_DIR, exist_ok=True); flat = {}; meta = {}
    for tag, rd in F11b.items():
        sk = _safe_key(tag)
        meta[tag] = dict(gid=rd['gid'], regime=rd['regime'], A_rho=rd['A_rho'],
                         chi_A_mean=rd['chi_A_mean'], alpha_th=rd['alpha_th'],
                         alpha_bf=(rd['alpha_bf'] if np.isfinite(rd['alpha_bf']) else None))
        flat[f'{sk}__t'] = rd['t']
        for n in (1, 2):
            for key, arr in rd['modes'][n].items():
                if arr is None: continue
                flat[f'{sk}__m{n}__{key}'] = np.asarray(arr)
    np.savez_compressed(os.path.join(OUTPUT_DIR, npz), **flat)
    with open(os.path.join(OUTPUT_DIR, js), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  -> F11b arrays saved: {npz}, {js}')

def load_F11b_data(npz='F11b_data.npz', js='F11b_data.json'):
    with open(os.path.join(OUTPUT_DIR, js)) as f:
        meta = json.load(f)
    z = np.load(os.path.join(OUTPUT_DIR, npz)); F11b = {}
    for tag, m in meta.items():
        sk = _safe_key(tag); modes = {1: {}, 2: {}}
        for n in (1, 2):
            for key in ('A_meas', 'P_const', 'P_bf', 'P_chi2'):
                full = f'{sk}__m{n}__{key}'
                modes[n][key] = z[full] if full in z.files else None
        F11b[tag] = dict(gid=m['gid'], regime=m['regime'], A_rho=m['A_rho'],
                         chi_A_mean=m['chi_A_mean'], alpha_th=m['alpha_th'],
                         alpha_bf=(m['alpha_bf'] if m['alpha_bf'] is not None else np.nan),
                         t=z[f'{sk}__t'], modes=modes)
    return F11b

def plot_F11b_modes(F11b, output_name='F11b_erosion_modes.pdf',
                    use_poster_style=True, A_FLOOR=2e-3):
    if use_poster_style:
        setup_mpl_poster()
    fig, axes = _make_2x4(figsize=(22.0, 11.0), sharey=True)
    LS_MEAS_K1 = '-'; LS_MEAS_K2 = (0, (6, 2.5))
    LS_CONST = ':'; LS_BF = (0, (3, 1.5, 1, 1.5)); LS_CHI2 = (0, (5, 1.5, 1, 1.5))
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, rd) for tag, rd in F11b.items() if rd['gid'] == gid],
                      key=lambda x: x[1]['A_rho'])
        flagged = dict(const=False, bf=False, chi2=False); A_repr = None
        for tag, rd in runs:
            t = rd['t']; col = _color(rd['A_rho']); reg = rd['regime']
            md1 = rd['modes'][1]; md2 = rd['modes'][2]; A_repr = rd['A_rho']
            ax.semilogy(t, np.abs(md1['A_meas']) + A_FLOOR, color=col,
                        ls=LS_MEAS_K1, lw=4.0,
                        label=rf'$A_\rho={rd["A_rho"]:.3g}$, $k{{=}}1$')
            ax.semilogy(t, np.abs(md2['A_meas']) + A_FLOOR, color=col,
                        ls=LS_MEAS_K2, lw=3.0, alpha=0.85,
                        label=rf'$A_\rho={rd["A_rho"]:.3g}$, $k{{=}}2$')
            if reg == 'strong':
                if md1.get('P_const') is not None:
                    ax.semilogy(t, md1['P_const'], color=col, ls=LS_CONST, lw=2.6,
                                alpha=0.8,
                                label=(r'$\alpha_{\rm th}$' if not flagged['const'] else None))
                    flagged['const'] = True
                if md1.get('P_bf') is not None:
                    ax.semilogy(t, md1['P_bf'], color=col, ls=LS_BF, lw=2.6, alpha=0.9,
                                label=(rf'$\alpha_{{\rm flux}}={rd["alpha_bf"]:.3f}$'
                                       if not flagged['bf'] else None))
                    flagged['bf'] = True
            else:
                if md1.get('P_chi2') is not None:
                    ax.semilogy(t, md1['P_chi2'], color=col, ls=LS_CHI2, lw=2.6,
                                alpha=0.9,
                                label=(r'$\alpha_{\rm th}\chi_A^2$' if not flagged['chi2'] else None))
                    flagged['chi2'] = True
        ax.axhline(0.5, color='0.40', ls=':', lw=1.6, alpha=0.6)
        ax.set_xlabel(r'$t/t_A$'); ax.set_xlim(left=0); ax.set_ylim(A_FLOOR*0.5, 2.0)
        ax.set_title(panel_title(gid, A_repr))
        ax.grid(True, which='both', alpha=0.25); _tick_both(ax)
        ax.legend(loc='lower left', framealpha=0.92, handlelength=2.4, fontsize=13)
        _add_regime_badge(ax, gid, loc='upper right')
    for ax in axes[:4]:
        ax.set_ylabel('')
    axes[0].set_ylabel(r'$A_\rho(t)/A_{\rho,k=1}(0)$')
    axes[4].set_ylabel(r'$A_\rho(t)/A_{\rho,k=1}(0)$')
    fig.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=300, bbox_inches='tight')
    plt.close(fig); print(f'  -> {os.path.join(OUTPUT_DIR, output_name)}')
    if use_poster_style:
        restore_mpl_default()

def fig_theory_vs_measurement(all_data, report_lines):
    report_lines += ['\n' + '=' * 70,
                     'F11b k=1/k=2 EROSION  (flux-calibrated alpha)', '=' * 70]
    F11b = compute_F11b_data(all_data, save=True)
    for tag, rd in sorted(F11b.items()):
        report_lines.append(f'  {tag:24s}  [{rd["regime"]:>9s}]  '
                            f'a_flux={rd["alpha_bf"]:.4f}  <chi_A>={rd["chi_A_mean"]:.2f}')
    plot_F11b_modes(F11b)


# =============================================================================
#  F12 — integrated flux (2x4)
# =============================================================================

def fig_integrated_flux_all_groups(all_data):
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']
            ax.plot(d['t'], integrate_abs_flux(d), color=_color(m['A_rho']),
                    lw=2.4, label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$\int|\Gamma_\rho|\,\mathrm{d}y$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F12_integrated_flux.pdf')


# =============================================================================
#  F14 — slaved closure (2x4)
# =============================================================================


def fig_slaved_closure(all_data):
    """Core flux-closure figure: measured flux versus fixed- and dynamic-l_perp
    predictions.  Each panel shows a single representative run per group, with
    only three curves: measured Gamma_rho, expected fixed-l_perp, and expected
    computed-l_perp."""
    fig, axes = _make_2x4(figsize=(19.0, 8.8), sharey=False)
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        if not runs:
            ax.axis('off')
            continue

        # Representative run: highest A_rho in the group (maximal signal)
        tag, d = runs[-1]
        m = d['meta']
        t = d['t']
        a_th = alpha_theory_imbalance(float(np.nanmean(d['sigma_c'])))

        gamma_meas = np.sqrt(np.nanmean(d['Gamma_rho']**2, axis=1))
        grad_rms = np.sqrt(np.nanmean(d['drho0_dy']**2, axis=1))

        l_fixed = d['l_perp_y'][0] if np.isfinite(d['l_perp_y'][0]) and d['l_perp_y'][0] > 0 \
            else 1.0 / max(m['k_perp_phys'], 1e-30)
        pred_fixed = a_th * d['zplus_rms'] * l_fixed * grad_rms
        pred_dyn = a_th * d['zplus_rms'] * np.where(d['l_perp_y'] > 0, d['l_perp_y'], np.nan) * grad_rms

        ax.plot(t, gamma_meas, color='k', lw=2.6, label=r'Measured $\Gamma_\rho$')
        ax.plot(t, pred_fixed, color='#1f77b4', ls='--', lw=2.2,
                label=r'Expected (fixed $\ell_\perp$)')
        ax.plot(t, pred_dyn, color='#d62728', ls='-', lw=2.2,
                label=r'Expected (computed $\ell_\perp$)')

        ax.set_xlabel(r'$t/t_A$')
        ax.set_ylabel(r'$\Gamma_\rho$')
        ax.set_xlim(left=0)
        ax.set_title(panel_title(gid, m['A_rho']))
        _tick_both(ax)
        ax.grid(True, alpha=0.22)
        ax.legend(loc='best', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F14_slaved_closure.pdf')

def fig_slaved_closure_measured(all_data):
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        if not runs:
            ax.axis('off'); continue
        A_repr = runs[-1][1]['meta']['A_rho']
        for tag, d in runs:
            m = d['meta']
            ax.plot(d['t'], d['drho_over_rho0_rms'], color=_color(m['A_rho']),
                    lw=2.6, label=rf'$A_\rho={m["A_rho"]:.3g}$')
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$(\delta\rho/\rho_0)_{\rm rms}$')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F14b_slaved_closure_measured.pdf')


# =============================================================================
#  F15 — K_perp vs k_diss (2x4)
# =============================================================================

def fig_Kperp_vs_dissipation(all_data):
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        kdiss = k_diss_of(gid)
        ax.axhline(kdiss, color='k', ls='--', lw=1.5, alpha=0.7,
                   label=rf'$k_{{\rm diss}}$')
        ax.axhline(GROUP_META[gid]['kappa_perp']*2*np.pi, color='0.5', ls=':',
                   lw=1.2, alpha=0.6, label=r'$k_{\perp,0}$')
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']; col = _color(m['A_rho']); t = d['t']
            ax.semilogy(t, Kperp_phase_mixing(d), color=col, ls='-', lw=2.2)
            ilpy = np.where(d['l_perp_y'] > 0, 1./d['l_perp_y'], np.nan)
            ax.semilogy(t, ilpy, color=col, ls='-.', lw=1.4, alpha=0.7)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$k_\perp\;[L_y^{-1}]$')
        ax.set_xlim(left=0); ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='lower right', fontsize=11)
        _add_regime_badge(ax, gid, loc='upper right')
    savefig(fig, 'F15_Kperp_vs_kdiss.pdf')


# =============================================================================
#  F16 — slaved ratio R(t) (2x4)
# =============================================================================

def fig_slaved_ratio(all_data, report_lines):
    PROJ = 1.0/np.sqrt(2.0); fig, axes = _make_2x4()
    report_lines += ['\n' + '=' * 70, 'SLAVED RATIO R(t)', '=' * 70]
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        runs = sorted([(tag, d) for tag, d in all_data.items()
                       if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0],
                      key=lambda x: x[1]['meta']['A_rho'])
        A_repr = None
        for tag, d in runs:
            m = d['meta']; A_repr = m['A_rho']
            lps = np.where(d['l_perp_full'] > 0, d['l_perp_full'], 0.0)
            R = PROJ*lps*d['Frho_rms_y']/(d['drho_over_rho0_rms']+1e-30)
            ax.plot(d['t'], R, color=_color(m['A_rho']), lw=2.4,
                    label=rf'$A_\rho={m["A_rho"]:.3g}$')
            Rf = R[np.isfinite(R)]
            report_lines.append(f'  {tag:24s}  <R>={float(np.nanmean(Rf)) if Rf.size else np.nan:.3f}')
        ax.axhline(1.0, color='k', ls='--', lw=1.4, alpha=0.65)
        ax.axhspan(0.8, 1.2, color='grey', alpha=0.10)
        ax.set_xlabel(r'$t/t_A$'); ax.set_ylabel(r'$R(t)$')
        ax.set_xlim(left=0); ax.set_ylim(0, 3.0)
        ax.set_title(panel_title(gid, A_repr))
        _tick_both(ax); ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=11)
        _add_regime_badge(ax, gid, loc='upper left')
    savefig(fig, 'F16_slaved_ratio.pdf')


# =============================================================================
#  F17 / F18 — cross-coherence and cross-phase spectra (2x4)
# =============================================================================


def _spec_accumulators(rho_slice, vy_slice, Ly, Lz, bin_edges):
    NZ, NY = rho_slice.shape
    Frho = np.fft.rfft2(rho_slice)
    Fvy = np.fft.rfft2(vy_slice)
    ky = np.fft.rfftfreq(NY, d=Ly / NY) * 2.0 * np.pi
    kz = np.fft.fftfreq(NZ, d=Lz / NZ) * 2.0 * np.pi
    KY, KZ = np.meshgrid(ky, kz)
    KP = np.sqrt(KY**2 + KZ**2).ravel()

    P_rr = (np.abs(Frho)**2).ravel()
    P_vv = (np.abs(Fvy)**2).ravel()
    P_rv = (Frho * np.conj(Fvy)).ravel()

    s_rr, _ = np.histogram(KP, bins=bin_edges, weights=P_rr)
    s_vv, _ = np.histogram(KP, bins=bin_edges, weights=P_vv)
    s_re, _ = np.histogram(KP, bins=bin_edges, weights=P_rv.real)
    s_im, _ = np.histogram(KP, bins=bin_edges, weights=P_rv.imag)
    cnt, _ = np.histogram(KP, bins=bin_edges)
    return s_rr, s_vv, s_re, s_im, cnt


def _weighted_shell_phase_and_r(s_rr, s_vv, s_re, s_im, coh_min=0.12, power_frac=1e-3):
    """Return a power-weighted shell phase, signed correlation coefficient r,
    coherence, and the shell weight used for masking.

    The shell phase is computed from the complex cross-spectrum summed over all
    modes in the shell and unwrapped across k_perp to suppress +/-pi wrapping.
    """
    cross = s_re + 1j * s_im
    power = np.sqrt(np.maximum(s_rr, 0.0) * np.maximum(s_vv, 0.0))
    denom = np.maximum(s_rr * s_vv, 1e-60)
    coh = (np.abs(cross)**2) / denom

    phase_raw = np.angle(cross)
    phase = np.full_like(phase_raw, np.nan, dtype=np.float64)
    r = np.full_like(phase_raw, np.nan, dtype=np.float64)

    finite = np.isfinite(phase_raw) & np.isfinite(power) & (power > 0)
    if np.any(finite):
        pcut = power_frac * np.nanmax(power[finite])
        keep = finite & (power >= pcut) & (coh >= coh_min)
        idx = np.where(keep)[0]
        if idx.size >= 2:
            phase[idx] = np.unwrap(phase_raw[idx])
        elif idx.size == 1:
            phase[idx] = phase_raw[idx]
        den = np.sqrt(np.maximum(s_rr * s_vv, 1e-60))
        r[idx] = np.real(cross[idx]) / den[idx]

    return phase, r, coh, power

def _select_target_runs(all_data, target_arho):
    tr = {}
    for tag, d in all_data.items():
        gid = d['meta']['gid']
        if d['meta']['A_rho'] <= 0: continue
        if gid in tr:
            if abs(d['meta']['A_rho']-target_arho) < abs(tr[gid]['meta']['A_rho']-target_arho):
                tr[gid] = d; tr[gid+1000] = tag
        else:
            tr[gid] = d; tr[gid+1000] = tag
    return tr

def _spec_setup(snaps, gid):
    d0 = aread.athdf(snaps[0]); rho0 = np.asarray(d0['rho'], dtype=np.float64)
    NZ, NY, NX = rho0.shape
    try:
        Ly = float(d0['RootGridX2'][1]) - float(d0['RootGridX2'][0])
    except Exception:
        Ly = GROUP_META[gid]['Ly']
    try:
        Lz = float(d0['RootGridX3'][1]) - float(d0['RootGridX3'][0])
    except Exception:
        Lz = GROUP_META[gid]['Lz']
    del rho0, d0; gc.collect()
    k_min = 2*np.pi/max(Ly, Lz); k_max = np.pi*min(NY/Ly, NZ/Lz)
    N_bins = max(40, min(NY//4, 80))
    edges = np.logspace(np.log10(k_min*0.9), np.log10(k_max*1.1), N_bins+1)
    centres = 0.5*(edges[:-1]+edges[1:])
    return NX, NY, NZ, Ly, Lz, N_bins, edges, centres

def fig_cross_coherence_spectrum(cat, all_data, target_arho=0.05,
                                 n_times=6, nx_planes=8):
    tr = _select_target_runs(all_data, target_arho)
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        if gid not in tr:
            ax.axis('off'); continue
        d = tr[gid]; tag = tr[gid+1000]; snaps = cat[tag]['snapshots']
        m = d['meta']; t_arr = d['t']; NT = len(snaps)
        if NT == 0:
            ax.axis('off'); continue
        idx = np.linspace(0, NT-1, min(n_times, NT), dtype=int)
        norm_t = Normalize(vmin=t_arr[0], vmax=t_arr[-1]); cmap_t = plt.cm.viridis
        kp0 = m['k_perp_phys']; kdiss = k_diss_of(gid)
        try:
            NX, NY, NZ, Ly, Lz, N_bins, edges, centres = _spec_setup(snaps, gid)
        except Exception as exc:
            print(f'  WARN F17 {tag}: {exc}'); ax.axis('off'); continue
        for si in idx:
            try:
                raw = aread.athdf(snaps[si])
            except Exception:
                continue
            rho3d = np.asarray(raw['rho'], dtype=np.float64)
            vy3d = np.asarray(raw['vel2'], dtype=np.float64); del raw; gc.collect()
            r0 = np.mean(rho3d, axis=(0, 2), keepdims=True)
            dvy = vy3d - np.mean(vy3d, axis=(0, 2), keepdims=True)
            drho = (rho3d - r0)/(r0+1e-30); del rho3d, vy3d; gc.collect()
            ixs = np.linspace(0, NX-1, min(nx_planes, NX), dtype=int)
            s_rr = np.zeros(N_bins); s_vv = np.zeros(N_bins)
            s_re = np.zeros(N_bins); s_im = np.zeros(N_bins); cnt = np.zeros(N_bins, dtype=np.int64)
            for ix in ixs:
                rr, vv, re, im, c = _spec_accumulators(drho[:, :, ix], dvy[:, :, ix], Ly, Lz, edges)
                s_rr += rr; s_vv += vv; s_re += re; s_im += im; cnt += c
            del drho, dvy; gc.collect()
            cs = np.maximum(cnt, 1)
            Prr = s_rr/cs; Pvv = s_vv/cs; Prv = (s_re+1j*s_im)/cs
            coh = (np.abs(Prv)**2)/np.maximum(Prr*Pvv, 1e-60)
            ax.plot(centres/kp0, coh, color=cmap_t(norm_t(t_arr[si])), lw=1.8,
                    alpha=0.92, label=(rf'$t={t_arr[si]:.1f}$' if si == idx[0] else None))
        ax.axvline(1.0, color='k', ls='--', lw=1.0, alpha=0.55)
        ax.axvline(kdiss/kp0, color='r', ls='--', lw=1.0, alpha=0.55)
        ax.set_xscale('log'); ax.set_xlabel(r'$k_\perp/k_{\perp,0}$')
        ax.set_ylabel(r'$C(k_\perp)$'); ax.set_xlim(left=0.5); ax.set_ylim(-0.05, 1.10)
        ax.set_title(panel_title(gid, m['A_rho']))
        _tick_both(ax); ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')
    tmax = max(d['t'].max() for d in all_data.values() if d['meta']['A_rho'] > 0)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(0, tmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.015, pad=0.01)
    cb.set_label(r'$t/t_A$')
    savefig(fig, f'F17_coherence_Arho{target_arho:.3f}.pdf')


def fig_cross_phase_spectrum(cat, all_data, target_arho=0.05, n_times=3,
                             nx_planes=8, coh_min=0.12):
    """Weighted/unwrapped phase spectrum plus signed correlation coefficient r.
    The phase is obtained from the power-weighted complex cross-spectrum shell
    sum and unwrapped across k_perp so that +/-pi wrapping does not dominate."""
    tr = _select_target_runs(all_data, target_arho)
    fig, axes = _make_2x4()
    for ax, gid in zip(axes, ACTIVE_GIDS_2x4):
        if gid not in tr:
            ax.axis('off')
            continue
        d = tr[gid]
        tag = tr[gid + 1000]
        snaps = cat[tag]['snapshots']
        m = d['meta']
        t_arr = d['t']
        NT = len(snaps)
        if NT == 0:
            ax.axis('off')
            continue

        idx = np.linspace(0, NT - 1, min(n_times, NT), dtype=int)
        norm_t = Normalize(vmin=t_arr[0], vmax=t_arr[-1])
        cmap_t = plt.cm.viridis
        kp0 = m['k_perp_phys']
        kdiss = k_diss_of(gid)
        try:
            NX, NY, NZ, Ly, Lz, N_bins, edges, centres = _spec_setup(snaps, gid)
        except Exception as exc:
            print(f'  WARN F18 {tag}: {exc}')
            ax.axis('off')
            continue

        axr = ax.twinx()
        axr.set_ylabel(r'$r(k_\perp)$')
        axr.set_ylim(-1.05, 1.05)
        axr.axhline(0.0, color='0.45', lw=0.8, alpha=0.35)

        for si in idx:
            try:
                raw = aread.athdf(snaps[si])
            except Exception:
                continue
            rho3d = np.asarray(raw['rho'], dtype=np.float64)
            vy3d = np.asarray(raw['vel2'], dtype=np.float64)
            del raw
            gc.collect()

            r0 = np.mean(rho3d, axis=(0, 2), keepdims=True)
            dvy = vy3d - np.mean(vy3d, axis=(0, 2), keepdims=True)
            drho = (rho3d - r0) / (r0 + 1e-30)
            del rho3d, vy3d
            gc.collect()

            ixs = np.linspace(0, NX - 1, min(nx_planes, NX), dtype=int)
            s_rr = np.zeros(N_bins)
            s_vv = np.zeros(N_bins)
            s_re = np.zeros(N_bins)
            s_im = np.zeros(N_bins)
            cnt = np.zeros(N_bins, dtype=np.int64)

            for ix in ixs:
                rr, vv, re, im, c = _spec_accumulators(drho[:, :, ix], dvy[:, :, ix],
                                                       Ly, Lz, edges)
                s_rr += rr
                s_vv += vv
                s_re += re
                s_im += im
                cnt += c

            del drho, dvy
            gc.collect()

            cs = np.maximum(cnt, 1)
            Prr = s_rr / cs
            Pvv = s_vv / cs
            Prv = (s_re + 1j * s_im) / cs

            phase, r_shell, coh, power = _weighted_shell_phase_and_r(
                Prr, Pvv, Prv.real, Prv.imag, coh_min=coh_min
            )

            mask = np.isfinite(phase) & np.isfinite(r_shell)
            if mask.sum() < 1:
                continue

            x = centres / kp0
            ax.plot(x[mask], phase[mask],
                    color=cmap_t(norm_t(t_arr[si])), lw=1.8, alpha=0.92,
                    label=(rf'$t={t_arr[si]:.1f}$' if si == idx[0] else None))
            axr.plot(x[mask], r_shell[mask],
                     color=cmap_t(norm_t(t_arr[si])), lw=1.2, ls=':',
                     alpha=0.85)

        ax.axhline(0.0, color='0.45', lw=0.8, alpha=0.45)
        ax.axhline(np.pi / 2, color='0.45', ls='--', lw=0.8, alpha=0.35)
        ax.axhline(-np.pi / 2, color='0.45', ls='--', lw=0.8, alpha=0.35)
        ax.axvline(1.0, color='k', ls='--', lw=1.0, alpha=0.55)
        ax.axvline(kdiss / kp0, color='r', ls='--', lw=1.0, alpha=0.55)

        ax.set_xscale('log')
        ax.set_xlabel(r'$k_\perp/k_{\perp,0}$')
        ax.set_ylabel(r'$\phi(k_\perp)$ [rad]')
        ax.set_xlim(left=0.5)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_title(panel_title(gid, m['A_rho']))
        _tick_both(ax)
        ax.grid(True, which='both', alpha=0.22)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
        _add_regime_badge(ax, gid, loc='upper left')

    tmax = max(d['t'].max() for d in all_data.values() if d['meta']['A_rho'] > 0)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(0, tmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), location='right', fraction=0.015, pad=0.01)
    cb.set_label(r'$t/t_A$')
    savefig(fig, f'F18_phase_Arho{target_arho:.3f}.pdf')

def compute_bridge_row(tag, data):
    m = data['meta']
    if m['A_rho'] == 0:
        return None
    chi_mean = float(np.nanmean(chi_A(data)))
    sigma_mean = float(np.nanmean(data['sigma_c']))
    fit = regress_alpha_from_flux(data)
    lpy = np.where(data['l_perp_y'] > 0, data['l_perp_y'], np.nan)
    eta_mean = float(np.nanmean(fit['alpha_fit'] * data['zplus_rms'] * lpy))
    lpz = np.where(data['l_perp_z'] > 0, data['l_perp_z'], np.nan)
    aniso = float(np.nanmean(lpy / lpz))
    K_ratio = float(np.nanmean(Kperp_phase_mixing(data) / k_diss_of(m['gid'])))
    r_rms = float(np.sqrt(np.nanmean(data['rho_vy_corr']**2)))
    return dict(tag=tag, gid=int(m['gid']), regime=m['regime'],
                square=bool(m['square']), A_rho=float(m['A_rho']),
                chi_A_mean=chi_mean, sigma_c_mean=sigma_mean, eta_eff_mean=eta_mean,
                aniso_y_over_z=aniso, Kperp_over_kdiss=K_ratio,
                rho_vy_corr_rms=r_rms,
                alpha_fit=fit['alpha_fit'], alpha_err=fit['alpha_err'],
                residual_rms=fit['residual_rms'], R2=fit['r_squared'])

def build_bridge_table(all_data):
    rows = []
    for tag in sorted(all_data.keys()):
        r = compute_bridge_row(tag, all_data[tag])
        if r is not None:
            rows.append(r)
    return rows


def save_bridge_table_csv(rows, path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, 'bridge_table.csv')
    keys = ['tag', 'regime', 'square', 'gid', 'A_rho', 'chi_A_mean', 'sigma_c_mean',
            'eta_eff_mean', 'aniso_y_over_z', 'Kperp_over_kdiss', 'rho_vy_corr_rms',
            'alpha_fit', 'alpha_err', 'residual_rms', 'R2']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  -> bridge table CSV: {path}')


def save_bridge_table_latex(rows, path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, 'bridge_table.tex')
    by = {'strong': [], 'weak': [], 'reference': []}
    for r in rows:
        by[r['regime']].append(r)
    L = [r'\begin{table*}', r'\centering',
         (r'\caption{Per-run flux-calibrated transport summary. '
          r'$\alpha$ is the direct flux-regression coefficient of '
          r'$\Gamma_\rho=-\alpha\,z^+_{\rm rms}\ell_{\perp,y}\partial_y\rho_0$.}'),
         r'\label{tab:bridge}', r'\begin{tabular}{l c c c c c c c}', r'\hline\hline',
         (r'Run & $A_\rho$ & $\langle\chi_A\rangle$ & $\langle\sigma_c\rangle$ & '
          r'$\langle\eta_{\rm eff}\rangle$ & $\langle\ell_{\perp,y}/\ell_{\perp,z}\rangle$ & '
          r'$\langle|r_{\rho v_y}|\rangle$ & $\alpha_{\rm eff}$ \\'),
         r'\hline']
    for reg in ('strong', 'weak', 'reference'):
        for r in by[reg]:
            tg = r['tag'].replace('_', r'\_')
            L.append(
                f'{tg} & {r["A_rho"]:.3g} & {r["chi_A_mean"]:.2f} & '
                f'{r["sigma_c_mean"]:.3f} & {r["eta_eff_mean"]:.3e} & '
                f'{r["aniso_y_over_z"]:.2f} & {r["rho_vy_corr_rms"]:.3f} & '
                f'${r["alpha_fit"]:.3f}\\pm{r["alpha_err"]:.3f}$ \\\\'
            )
        if by[reg]:
            L.append(r'\hline')
    L += [r'\hline', r'\end{tabular}', r'\end{table*}']
    with open(path, 'w') as f:
        f.write('\n'.join(L))
    print(f'  -> bridge table LaTeX: {path}')

def write_regime_summary(all_data, report_lines):
    report_lines += ['\n' + '=' * 70,
                     'REGIME SUMMARY  alpha_flux vs Lorentzian prediction', '=' * 70]
    for name, gids in (('STRONG', STRONG_GIDS), ('WEAK', WEAK_GIDS),
                       ('REFERENCE', REFERENCE_GIDS)):
        report_lines.append(f'\n  --- {name} ---')
        for gid in gids:
            for tag, d in sorted(all_data.items()):
                if d['meta']['gid'] != gid or d['meta']['A_rho'] == 0:
                    continue
                af, _ = alpha_flux(d); cm = float(np.nanmean(chi_A(d)))
                ap = (1/8.)*cm**2/(1+cm**2)
                re = (af-ap)/ap if ap > 0 and np.isfinite(af) else np.nan
                report_lines.append(f'  {tag:24s}  chi_A={cm:6.2f}  '
                                    f'a_flux={af:.4f}  a_pred={ap:.4f}  rel_err={re:+.1%}')

def write_report(report_lines):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'unified_diagnostics_report_v12.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'  Report -> {path}')


# =============================================================================
#  MAIN
# =============================================================================


def main():
    print(__doc__)
    setup_mpl()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = [
        'Unified Density Transport Diagnostics — v12 (weighted phase + reduced closure)',
        '=' * 70,
        f'Strong gids {STRONG_GIDS} (S1..S4) -> top row of 2x4 grids',
        f'Weak   gids {WEAK_GIDS} (W1..W4) -> bottom row of 2x4 grids',
        f'Reference gid {REFERENCE_GIDS} -> overlay-only + F02b',
        'Primary alpha source : regress_alpha_from_flux (v9 engine)',
        'Theory curve         : alpha_eff = (1/8) chi_A^2/(1+chi_A^2)',
        f'X_DISS={X_DISS}, N_y={NY_GRID}, v_A^2/c_s^2={VA2_OVER_CS2:.3f}',
        f'Output dir : {OUTPUT_DIR}',
        '=' * 70,
        ''
    ]

    print('\nLoading simulations...')
    all_data, cat = load_all_runs()
    if not all_data:
        print('ERROR: no data loaded.')
        return

    # lightweight run-level r diagnostic report
    for tag, d in sorted(all_data.items()):
        if d['meta']['A_rho'] <= 0:
            continue
        report_lines.append(
            f'  {tag:24s}  r_rho_vy,rms={np.sqrt(np.nanmean(d["rho_vy_corr"]**2)):.3f}  '
            f'chi_A={float(np.nanmean(chi_A(d))):.2f}'
        )

    print('\nGenerating figures...')
    fig_zplus_overlay(all_data)
    fig_rho_waterfalls(all_data)
    fig_reference_waterfall(all_data)
    fig_rho_waterfalls_full(all_data)
    fig_lperp_comparison(all_data)
    fig_alpha_dynamic(all_data, report_lines)
    fig_chi_A(all_data, report_lines)
    fig_flux_collapse(all_data, report_lines)
    fig_flux_regression_clouds(all_data)
    fig_alpha_vs_chi(all_data, report_lines)
    fig_morphology_diagnostics(all_data)
    fig_gradient_thickness(all_data)
    fig_erosion_raw(all_data)
    fig_theory_vs_measurement(all_data, report_lines)
    fig_integrated_flux_all_groups(all_data)
    fig_slaved_closure(all_data)
    fig_Kperp_vs_dissipation(all_data)
    fig_slaved_ratio(all_data, report_lines)
    fig_cross_coherence_spectrum(cat, all_data, target_arho=0.05, n_times=6, nx_planes=8)
    fig_cross_phase_spectrum(cat, all_data)
    write_regime_summary(all_data, report_lines)
    write_report(report_lines)
    print(f'\nAll outputs in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

