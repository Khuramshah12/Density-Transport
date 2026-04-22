"""
rmhd_closure_diagnostic_v7.py
==============================
Specialized diagnostic comparing the measured k=1 density amplitude erosion
against the slaved-limit Fickian closure prediction for all simulation groups.

Physical model
--------------
The slaved-limit theory predicts that the k_y=1 Fourier amplitude of the
xz-averaged background density evolves as:

    A_rho(t) = A_rho(0) * exp( -K_y^2 * I(t) )

where the cumulative diffusivity integral is:

    I(t) = integral_0^t  eta_turb(t') dt'

    eta_turb(t') = alpha * z+_rms(t') * l_perp(t')

and the theoretical closure sets alpha = 1/4 = 0.25.

If the prediction does not match the simulation, two sources of discrepancy
are possible:
  (a) The prefactor alpha deviates from 0.25 — either because the isotropy
      and slaving assumptions are imperfect, or because finite-beta slow-mode
      contributions are non-negligible.
  (b) The functional form exp(-K_y^2 I) is correct but l_perp is mis-measured
      (e.g., the geometric mean of l_y and l_z over- or under-estimates the
      flux-generating correlation length).

This script isolates case (a) by performing a least-squares calibration of
alpha for each group and run, and reports whether the elongated box (Group 3,
L_x=5) shows systematically different efficiency from the cubic groups.

Outputs
-------
  fig_closure_comparison.pdf   — four-panel figure (one per group)
  closure_alpha_calibration.pdf — scatter plot of calibrated alpha vs run parameters
  closure_diagnostics.txt       — text report of divergence times and alpha values
"""

import os
import gc
import glob
import sys
import warnings
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz   # scipy < 1.8

warnings.filterwarnings('ignore')

sys.path.insert(1, '/nesi/nobackup/uoo02637/khurram/density_transport/'
                   'Initial Conditions/')
import athena_read as aread


# =============================================================================
#  CONFIGURATION
# =============================================================================

BASE = ('/home/abbkh891/00_nesi_projects/uoo02637_nobackup/khurram/'
        'density_transport/500_k15/')

OUTPUT_SPECS = [
    (0.00,  BASE + 'rho0/output/',          0),
    (0.05,  BASE + 'rho0_05/output/',       0),
    (0.10,  BASE + 'rho0_1/output/',        0),
    (0.05,  BASE + 'rho0_05A02kp2/output/', 1),
    (0.10,  BASE + 'rho0_1A02kp2/output/',  1),
    (0.00,  BASE + 'rho0_zp01/output/',     2),
    (0.05,  BASE + 'rho0_05zp01/output/',   2),
    (0.10,  BASE + 'rho0_1zp01/output/',    2),
    (0.00,  BASE + 'rho0_Lx5/output/',      3),
    (0.025, BASE + 'rho0_025_Lx5/output/',  3),
    (0.05,  BASE + 'rho0_05_Lx5/output/',   3),
    (0.10,  BASE + 'rho0_1_Lx5/output/',    3),
]

GROUP_META = {
    0: dict(k_par_mode=1, k_perp_mode=5, z0p=0.2,
            Lx=1.0, Ly=1.0, Lz=1.0,
            ls='-',  lw=2.0,
            label=r'Group 0: $k_\parallel=1,\;z_0^+=0.2$ (cubic)'),
    1: dict(k_par_mode=2, k_perp_mode=5, z0p=0.2,
            Lx=1.0, Ly=1.0, Lz=1.0,
            ls='--', lw=2.0,
            label=r'Group 1: $k_\parallel=2,\;z_0^+=0.2$ (cubic)'),
    2: dict(k_par_mode=1, k_perp_mode=5, z0p=0.1,
            Lx=1.0, Ly=1.0, Lz=1.0,
            ls=':',  lw=1.8,
            label=r'Group 2: $k_\parallel=1,\;z_0^+=0.1$ (cubic)'),
    3: dict(k_par_mode=1, k_perp_mode=5, z0p=0.2,
            Lx=5.0, Ly=1.0, Lz=1.0,
            ls='-.', lw=2.0,
            label=r'Group 3: $k_\parallel^{\rm phys}=2\pi/5,\;z_0^+=0.2$ ($L_x=5$)'),
}

ALPHA_THEORY = 0.25          # theoretical closure prefactor
FILE_PATTERN = 'from_array.out.*.athdf'
OUTPUT_DIR   = BASE + 'closure_diagnostic_v7/'

# Divergence threshold: flag when |measured - theory| / measured > this fraction
DIVERGENCE_THRESHOLD = 0.10   # 10 percent relative error

COLOR = {
    0.000: '#1f77b4',
    0.025: '#17becf',
    0.050: '#2ca02c',
    0.100: '#d62728',
}


# =============================================================================
#  MATPLOTLIB SETTINGS
# =============================================================================

mpl.rcParams.update({
    'text.usetex':          True,
    'font.family':          'serif',
    'font.serif':           ['Computer Modern Roman'],
    'mathtext.fontset':     'cm',
    'font.size':            11,
    'axes.labelsize':       12,
    'axes.titlesize':       10,
    'legend.fontsize':      8.5,
    'legend.framealpha':    0.90,
    'legend.edgecolor':     '0.60',
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.top':            True,
    'ytick.right':          True,
    'xtick.minor.visible':  True,
    'ytick.minor.visible':  True,
    'axes.linewidth':       0.9,
    'lines.linewidth':      1.8,
    'grid.alpha':           0.25,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.06,
    'figure.dpi':           120,
})

try:
    import subprocess
    subprocess.run(['latex', '--version'], capture_output=True, check=True)
except Exception:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif']  = ['Times New Roman', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'


# =============================================================================
#  HELPERS
# =============================================================================

def savefig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'  -> {path}')


def _color(arho):
    return COLOR.get(arho, '#888888')


def _tick_both(ax):
    ax.tick_params(which='both', direction='in', top=True, right=True)


# =============================================================================
#  SNAPSHOT PROCESSOR  (minimal — only what this script needs)
# =============================================================================

def _process_snapshot(fn, gid):
    """
    Load one snapshot and return the scalar diagnostics needed for the
    closure comparison: t, z+_rms (amplitude), lperp_comb, A_rho_k1.

    The correlation length lperp_comb = sqrt(lperp_y * lperp_z) uses
    the geometric mean of the y-direction ACF scale (which shortens as
    phase mixing builds fine y-structure) and the z-direction ACF scale
    (which reflects the turbulent cascade and is less contaminated by
    phase mixing).
    """
    d   = aread.athdf(fn)
    t   = float(d['Time'])
    gm  = GROUP_META[gid]

    rho = np.asarray(d['rho'],  dtype=np.float64)
    vy  = np.asarray(d['vel2'], dtype=np.float64)
    vz  = np.asarray(d['vel3'], dtype=np.float64)
    By  = np.asarray(d['Bcc2'], dtype=np.float64)
    Bz  = np.asarray(d['Bcc3'], dtype=np.float64)

    NZ, NY, NX = rho.shape
    xz = (0, 2)

    # Grid
    try:
        ymin = float(d['RootGridX2'][0])
        ymax = float(d['RootGridX2'][1])
    except Exception:
        ymin, ymax = 0.0, gm['Ly']
    Ly    = ymax - ymin
    y_arr = ymin + (np.arange(NY) + 0.5) * (Ly / NY)

    try:
        Lz = float(d['RootGridX3'][1]) - float(d['RootGridX3'][0])
    except Exception:
        Lz = gm['Lz']

    # Background density
    rho0_3d = np.mean(rho, axis=xz, keepdims=True)
    rho0_1d = rho0_3d[0, :, 0]

    # k=1 Fourier amplitude of background profile
    rho_mean = float(np.mean(rho0_1d))
    sin_k    = np.sin(2.0 * np.pi * y_arr / Ly)
    cos_k    = np.cos(2.0 * np.pi * y_arr / Ly)
    try:
        A_s = (2./Ly) * float(np.trapezoid((rho0_1d/rho_mean - 1.) * sin_k, y_arr))
        A_c = (2./Ly) * float(np.trapezoid((rho0_1d/rho_mean - 1.) * cos_k, y_arr))
    except AttributeError:
        A_s = (2./Ly) * float(np.trapz((rho0_1d/rho_mean - 1.) * sin_k, y_arr))
        A_c = (2./Ly) * float(np.trapz((rho0_1d/rho_mean - 1.) * cos_k, y_arr))
    A_rho_k1 = float(np.sqrt(A_s**2 + A_c**2))

    # Elsässer amplitude (z+ only — code units, 4pi absorbed into B)
    inv_sq = 1.0 / np.sqrt(rho0_3d + 1e-30)
    dvy = vy - np.mean(vy, axis=xz, keepdims=True)
    dvz = vz - np.mean(vz, axis=xz, keepdims=True)
    dBy = By - np.mean(By, axis=xz, keepdims=True)
    dBz = Bz - np.mean(Bz, axis=xz, keepdims=True)
    zp_y = dvy + dBy * inv_sq
    zp_z = dvz + dBz * inv_sq
    zplus_rms = float(np.sqrt(np.mean(zp_y**2 + zp_z**2)))

    del rho, vy, vz, By, Bz, dvy, dvz, dBy, dBz, inv_sq
    gc.collect()

    # Correlation lengths via FFT-based ACF (Wiener–Khinchin theorem)
    def _acf_len(field, axis, L):
        N    = field.shape[axis]
        lags = np.arange(N) * (L / N)
        f    = field - np.mean(field, axis=axis, keepdims=True)
        Pk   = np.mean(np.abs(np.fft.fft(f, axis=axis))**2,
                       axis=tuple(i for i in range(3) if i != axis)) / N**2
        ac   = np.real(np.fft.ifft(Pk))
        if ac[0] <= 0:
            return np.nan
        R  = ac / ac[0]
        neg = np.where(R[1:N//2] <= 0)[0]
        zc  = int(neg[0]) + 1 if len(neg) else N // 2
        try:
            return max(float(np.trapezoid(R[:zc+1], lags[:zc+1])), 0.)
        except AttributeError:
            return max(float(np.trapz(R[:zc+1], lags[:zc+1])), 0.)

    lperp_y = _acf_len(zp_y, axis=1, L=Ly)
    lperp_z = _acf_len(zp_y, axis=0, L=Lz)

    if (np.isfinite(lperp_y) and np.isfinite(lperp_z)
            and lperp_y > 0 and lperp_z > 0):
        lperp_comb = float(np.sqrt(lperp_y * lperp_z))
    else:
        lperp_comb = lperp_z if np.isfinite(lperp_z) else lperp_y

    del zp_y, zp_z
    gc.collect()

    return dict(t=t, Ly=Ly, A_rho_k1=A_rho_k1,
                zplus_rms=zplus_rms,
                lperp_y=lperp_y, lperp_z=lperp_z, lperp_comb=lperp_comb)


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_all_runs():
    """
    Iterate over OUTPUT_SPECS, process every snapshot, and return a dict
    of time-series arrays keyed by run tag.

    Returns
    -------
    all_data : dict
        all_data[tag] = {
            'meta'        : dict with A_rho, gid, group parameters,
            't'           : (NT,) array of simulation times,
            'A_rho'       : (NT,) array of k=1 Fourier amplitudes,
            'A_rho_norm'  : (NT,) array normalised by A_rho[0],
            'zplus_rms'   : (NT,) array of z+ amplitudes,
            'lperp_comb'  : (NT,) array of geometric-mean correlation lengths,
            'Ly'          : box y-length (float),
        }
    """
    all_data = {}

    for A_rho, folder, gid in OUTPUT_SPECS:
        tag   = f'g{gid}_Arho{A_rho:.3f}'
        snaps = sorted(glob.glob(os.path.join(folder, FILE_PATTERN)))
        if not snaps:
            print(f'  [SKIP] {tag}: no files found in {folder}')
            continue

        print(f'\nLoading {tag} ({len(snaps)} snapshots) ...')
        records = []
        for i, fn in enumerate(snaps):
            try:
                records.append(_process_snapshot(fn, gid))
                if (i + 1) % 20 == 0 or (i + 1) == len(snaps):
                    r = records[-1]
                    print(f'  {i+1}/{len(snaps)}  t={r["t"]:.2f}  '
                          f'z+={r["zplus_rms"]:.4f}  '
                          f'A_rho={r["A_rho_k1"]:.5f}  '
                          f'lperp={r["lperp_comb"]:.5f}')
            except Exception as exc:
                print(f'  WARN: skipped {os.path.basename(fn)}: {exc}')

        if not records:
            print(f'  ERROR: no valid records for {tag}.')
            continue

        records.sort(key=lambda r: r['t'])
        t          = np.array([r['t']          for r in records])
        A_rho_raw  = np.array([r['A_rho_k1']   for r in records])
        zplus_rms  = np.array([r['zplus_rms']   for r in records])
        lperp_comb = np.array([r['lperp_comb']  for r in records])
        Ly         = records[0]['Ly']

        A0         = A_rho_raw[0] if A_rho_raw[0] > 1e-12 else 1.0
        A_rho_norm = A_rho_raw / A0

        gm = GROUP_META[gid]
        all_data[tag] = dict(
            meta=dict(
                A_rho=A_rho, gid=gid,
                k_par_phys=gm['k_par_mode'] * 2.*np.pi / gm['Lx'],
                z0p=gm['z0p'], Lx=gm['Lx'], Ly=Ly,
                ls=gm['ls'], lw=gm['lw'],
                group_label=gm['label'],
            ),
            t=t,
            A_rho=A_rho_raw,
            A_rho_norm=A_rho_norm,
            zplus_rms=zplus_rms,
            lperp_comb=lperp_comb,
            Ly=Ly,
            A0=A0,
        )
        del records
        gc.collect()

    return all_data


# =============================================================================
#  THEORY: CUMULATIVE INTEGRAL AND PREDICTION
# =============================================================================

def build_prediction(data, alpha):
    """
    Compute the theoretical prediction:

        P(t) = exp( -K_y^2 * alpha * integral_0^t z+_rms(t') l_perp(t') dt' )

    using scipy.integrate.cumtrapz (or cumulative_trapezoid in scipy >= 1.8).

    Parameters
    ----------
    data  : dict for one run (must contain 't', 'zplus_rms', 'lperp_comb', 'Ly')
    alpha : diffusivity prefactor (theory: 0.25)

    Returns
    -------
    I_cum   : (NT,) cumulative integral of eta(t') = alpha * z+ * l_perp
    P       : (NT,) theoretical amplitude ratio exp(-K_y^2 * I_cum)
    eta_arr : (NT,) instantaneous diffusivity array
    """
    t         = data['t']
    zp        = data['zplus_rms']
    lp        = data['lperp_comb']
    Ly        = data['Ly']
    Ky        = 2.0 * np.pi / Ly

    # Instantaneous diffusivity
    lp_safe   = np.where(np.isfinite(lp) & (lp > 0), lp, 0.0)
    eta_arr   = alpha * zp * lp_safe

    # Cumulative integral starting at zero
    I_cum     = np.zeros_like(t)
    I_cum[1:] = cumtrapz(eta_arr, t)

    P         = np.exp(-Ky**2 * I_cum)
    return I_cum, P, eta_arr


def calibrate_alpha(data):
    """
    Find the best-fit alpha minimising the RMS error between
    the measured A_rho_norm and the theoretical prediction P(t; alpha).

    Only the time window where lperp_comb is finite and A_rho_norm > 0.02
    is used (avoids fitting in the noise floor).

    Returns
    -------
    alpha_best : float
    rms_best   : float (RMS of residuals at alpha_best)
    """
    A_meas = data['A_rho_norm']
    valid  = (np.isfinite(data['lperp_comb'])
              & (data['lperp_comb'] > 0)
              & (A_meas > 0.02))

    if valid.sum() < 3:
        return np.nan, np.nan

    def _rms(alpha):
        _, P, _ = build_prediction(data, alpha)
        res     = A_meas[valid] - P[valid]
        return float(np.sqrt(np.mean(res**2)))

    result = minimize_scalar(_rms, bounds=(0.001, 2.0), method='bounded',
                             options={'xatol': 1e-4})
    return float(result.x), float(result.fun)


def find_divergence_time(t, A_meas, P_theory, threshold=DIVERGENCE_THRESHOLD):
    """
    Return the earliest time at which the relative error between the measured
    amplitude and the theoretical prediction exceeds `threshold`.

    Relative error = |A_meas - P_theory| / max(A_meas, 1e-4).

    Returns np.nan if no such time is found.
    """
    rel_err = np.abs(A_meas - P_theory) / np.maximum(A_meas, 1e-4)
    # Smooth over 5 steps to avoid flagging transient spikes
    rel_err_s = uniform_filter1d(rel_err.astype(float), size=5)
    idx       = np.where(rel_err_s > threshold)[0]
    if len(idx) == 0:
        return np.nan
    return float(t[idx[0]])


# =============================================================================
#  FIG 1 — FOUR-PANEL CLOSURE COMPARISON (one panel per group)
# =============================================================================

def fig_closure_comparison(all_data):
    """
    Four panels (one per group).  Within each panel:
      - Upper sub-panel: measured A_rho(t)/A_rho(0) (solid) and the
        theoretical prediction with alpha=0.25 (dashed, same colour).
      - Lower sub-panel: residual = measured - theory.

    A run with A_rho = 0 is shown as a thin baseline in the upper panel to
    confirm that the closure should return P = 1 (no erosion without a gradient).
    """
    gids   = sorted(GROUP_META.keys())
    n_grps = len(gids)

    fig    = plt.figure(figsize=(6.5 * n_grps, 7.5))
    outer  = gridspec.GridSpec(1, n_grps, figure=fig,
                                wspace=0.38, left=0.07, right=0.97,
                                top=0.91, bottom=0.09)

    for col, gid in enumerate(gids):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_res  = fig.add_subplot(inner[1], sharex=ax_main)

        runs_in_group = {
            tag: d for tag, d in all_data.items()
            if d['meta']['gid'] == gid
        }
        if not runs_in_group:
            ax_main.text(0.5, 0.5, 'No data', transform=ax_main.transAxes,
                         ha='center', va='center')
            continue

        for tag, data in sorted(runs_in_group.items(),
                                key=lambda x: x[1]['meta']['A_rho']):
            A_rho  = data['meta']['A_rho']
            if A_rho == 0:
                # Baseline: theory always gives P=1; plot measured as thin line
                ax_main.plot(data['t'], data['A_rho_norm'],
                             color=_color(0.0), ls='-', lw=0.9, alpha=0.55,
                             label=r'$A_\rho=0$ (baseline)')
                continue

            t      = data['t']
            A_meas = data['A_rho_norm']
            _, P_th, _ = build_prediction(data, ALPHA_THEORY)
            residual   = A_meas - P_th
            color      = _color(A_rho)
            lbl_meas   = rf'$A_\rho={A_rho:.3g}$, meas.'
            lbl_pred   = rf'$A_\rho={A_rho:.3g}$, pred. ($\alpha=0.25$)'

            ax_main.semilogy(t, A_meas,
                             color=color, ls=data['meta']['ls'],
                             lw=data['meta']['lw'], label=lbl_meas)
            ax_main.semilogy(t, P_th,
                             color=color, ls='--', lw=1.0, alpha=0.70,
                             label=lbl_pred)
            ax_res.plot(t, residual,
                        color=color, ls=data['meta']['ls'],
                        lw=1.2, alpha=0.85)

        # Formatting — main panel
        ax_main.axhline(0.5, color='grey', ls=':', lw=0.8, alpha=0.5)
        ax_main.set_xlim(left=0)
        ax_main.set_ylabel(r'$A_\rho(t)\,/\,A_\rho(0)$')
        ax_main.set_title(GROUP_META[gid]['label'], fontsize=9.5, pad=5)
        ax_main.legend(fontsize=7, loc='lower left', ncol=1)
        _tick_both(ax_main)
        ax_main.grid(True, which='both', alpha=0.22)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Formatting — residual panel
        ax_res.axhline(0, color='k', ls='-', lw=0.7, alpha=0.6)
        ax_res.axhspan(-DIVERGENCE_THRESHOLD, DIVERGENCE_THRESHOLD,
                       color='grey', alpha=0.12,
                       label=rf'{int(DIVERGENCE_THRESHOLD*100)}\% band')
        ax_res.set_xlabel(r'$t\,/\,t_A$')
        ax_res.set_ylabel(r'Residual', fontsize=9)
        ax_res.legend(fontsize=7, loc='upper right')
        _tick_both(ax_res)
        ax_res.grid(True, alpha=0.20)

    fig.suptitle(
        r'Fickian closure test: measured $A_\rho(t)/A_\rho(0)$ (solid) '
        r'vs.\ theory $\exp(-K_y^2\int\alpha z^+\ell_\perp\,dt)$ (dashed), '
        r'$\alpha=0.25$',
        fontsize=11, y=0.975)
    savefig(fig, 'fig_closure_comparison.pdf')


# =============================================================================
#  FIG 2 — BEST-FIT ALPHA CALIBRATION
# =============================================================================

def fig_alpha_calibration(all_data, report_lines):
    """
    For each run with A_rho > 0, compute the best-fit alpha and plot it as
    a scatter with:
      x-axis : k_par_phys * A_rho  (proxy for the phase-mixing rate per unit
                                    density gradient)
      y-axis : alpha_best
      color  : A_rho
      marker : group (0=circle, 1=square, 2=triangle, 3=diamond)

    A dashed horizontal line marks the theoretical value alpha = 0.25.
    Systematic vertical offset between the groups indicates that the closure
    prefactor depends on parameters beyond A_rho * k_par — most likely on
    the anisotropy ratio k_perp/k_par or on the finite-beta slow-mode
    contribution to the compressive sector.
    """
    MARKERS = {0: 'o', 1: 's', 2: '^', 3: 'D'}

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    for tag, data in all_data.items():
        m = data['meta']
        if m['A_rho'] == 0:
            continue

        alpha_best, rms_best = calibrate_alpha(data)
        if not np.isfinite(alpha_best):
            continue

        x      = m['k_par_phys'] * m['A_rho']
        color  = _color(m['A_rho'])
        marker = MARKERS.get(m['gid'], 'o')

        ax.scatter(x, alpha_best, s=110, color=color,
                   marker=marker, edgecolors='k', lw=0.8, zorder=5)

        # Record for text report
        report_lines.append(
            f'{tag:<30}  A_rho={m["A_rho"]:.3f}  gid={m["gid"]}  '
            f'k_par_phys={m["k_par_phys"]:.4f}  '
            f'alpha_best={alpha_best:.4f}  rms={rms_best:.5f}')

    ax.axhline(ALPHA_THEORY, color='k', ls='--', lw=1.2, alpha=0.65,
               label=rf'Theory: $\alpha = {ALPHA_THEORY}$')
    ax.set_xlabel(r'$k_\parallel^{\rm phys}\cdot A_\rho$')
    ax.set_ylabel(r'Best-fit $\alpha$')
    ax.set_ylim(bottom=0)
    _tick_both(ax)
    ax.grid(True, alpha=0.22)

    # Legend: colors for A_rho, markers for groups
    a_vals   = sorted({d['meta']['A_rho'] for d in all_data.values()
                       if d['meta']['A_rho'] > 0})
    color_h  = [Patch(facecolor=_color(a),
                      label=rf'$A_\rho={a:.3g}$') for a in a_vals]
    marker_h = [Line2D([0],[0], marker=MARKERS[g], color='grey',
                       markersize=8, markeredgecolor='k', lw=0,
                       label=GROUP_META[g]['label'].split(':')[0])
                for g in sorted(MARKERS)]
    theory_h = [Line2D([0],[0], color='k', ls='--', lw=1.2,
                       label=rf'$\alpha={ALPHA_THEORY}$ (theory)')]
    ax.legend(handles=color_h + marker_h + theory_h,
              fontsize=8, ncol=2, loc='upper right')

    fig.suptitle(
        r'Best-fit closure prefactor $\alpha$ per run.  '
        r'Deviation from $\alpha=0.25$ quantifies the closure error.',
        fontsize=10, y=1.005)
    savefig(fig, 'closure_alpha_calibration.pdf')


# =============================================================================
#  FIG 3 — ALPHA COMPARISON: CUBIC vs ELONGATED
# =============================================================================

def fig_alpha_boxplot(all_data, report_lines):
    """
    Box-and-whisker summary comparing the distribution of calibrated alpha
    values across the four groups.  This directly tests whether the elongated
    box (Group 3, L_x=5) has systematically higher or lower transport
    efficiency than the cubic cases.

    A higher alpha in Group 3 would indicate that the reduced phase-mixing
    rate allows the density and velocity fluctuations to remain better
    correlated, so that each unit of diffusivity z+*l_perp translates into
    a larger cross-field flux.  A lower alpha would indicate that the
    extended transport window is offset by a reduction in the per-unit
    flux efficiency.
    """
    group_alphas = {gid: [] for gid in GROUP_META}

    for tag, data in all_data.items():
        m = data['meta']
        if m['A_rho'] == 0:
            continue
        alpha_best, _ = calibrate_alpha(data)
        if np.isfinite(alpha_best):
            group_alphas[m['gid']].append(alpha_best)

    gids      = sorted(k for k, v in group_alphas.items() if len(v) > 0)
    positions = np.arange(len(gids))
    labels    = [GROUP_META[g]['label'].replace(r'Group ', 'G')
                 for g in gids]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    parts = ax.violinplot([group_alphas[g] for g in gids],
                          positions=positions,
                          showmeans=True, showextrema=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#aec7e8')
        pc.set_edgecolor('#1f77b4')
        pc.set_alpha(0.7)

    # Overlay individual points
    for pos, gid in zip(positions, gids):
        alphas = group_alphas[gid]
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(alphas))
        ax.scatter(pos + jitter, alphas, s=55,
                   color='#1f77b4', edgecolors='k', lw=0.7, zorder=5)

    ax.axhline(ALPHA_THEORY, color='k', ls='--', lw=1.2, alpha=0.65,
               label=rf'Theory: $\alpha={ALPHA_THEORY}$')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'Best-fit $\alpha$')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    _tick_both(ax)
    ax.grid(True, axis='y', alpha=0.22)

    fig.suptitle(
        r'Distribution of best-fit $\alpha$ by group.  '
        r'Offset from $\alpha=0.25$ (dashed) measures closure error.',
        fontsize=10, y=1.005)
    savefig(fig, 'fig_alpha_boxplot.pdf')

    # Write group-level summary to report
    report_lines.append('\n' + '─'*60)
    report_lines.append('Group-level alpha summary')
    report_lines.append('─'*60)
    for gid in gids:
        vals = group_alphas[gid]
        if vals:
            report_lines.append(
                f'  Group {gid}: n={len(vals)}  '
                f'mean={np.mean(vals):.4f}  '
                f'std={np.std(vals):.4f}  '
                f'min={np.min(vals):.4f}  '
                f'max={np.max(vals):.4f}  '
                + ('** ELONGATED BOX **' if gid == 3 else ''))


# =============================================================================
#  ADVERSARIAL DIVERGENCE DIAGNOSTIC
# =============================================================================

def adversarial_divergence_check(all_data, report_lines):
    """
    For every run with A_rho > 0 and the theoretical alpha = 0.25, find the
    first time at which the relative prediction error exceeds
    DIVERGENCE_THRESHOLD.  Print a warning if the divergence begins before
    the phase-mixing time t_pm, which would indicate that the theory fails
    even in the early, well-resolved phase of the simulation.
    """
    report_lines.append('\n' + '─'*70)
    report_lines.append(
        f'Adversarial divergence check (threshold = '
        f'{int(DIVERGENCE_THRESHOLD*100)}% relative error)')
    report_lines.append('─'*70)

    for tag, data in sorted(all_data.items()):
        m = data['meta']
        if m['A_rho'] == 0:
            continue

        t      = data['t']
        A_meas = data['A_rho_norm']
        _, P_th, _ = build_prediction(data, ALPHA_THEORY)

        t_div = find_divergence_time(t, A_meas, P_th, DIVERGENCE_THRESHOLD)

        # Phase-mixing timescale for this run
        lp0    = data['lperp_comb'][0] if data['lperp_comb'][0] > 0 else np.nan
        # Approximate t_pm from the initial |dvA/dy| if available;
        # otherwise omit the comparison
        t_pm   = np.nan   # computed externally; stored in v6 timescales

        if np.isfinite(t_div):
            msg = (f'  {tag:<30}  divergence at t={t_div:.2f} t_A  '
                   f'(A_rho={m["A_rho"]:.3f}, gid={m["gid"]})')
            # Over- or under-prediction?
            idx_div = np.searchsorted(t, t_div)
            if idx_div < len(A_meas):
                diff = A_meas[idx_div] - P_th[idx_div]
                direction = 'THEORY OVER-PREDICTS' if diff > 0 else 'THEORY UNDER-PREDICTS'
                msg += f'  {direction}'
            print(msg)
            report_lines.append(msg)
        else:
            msg = (f'  {tag:<30}  no divergence detected  '
                   f'(A_rho={m["A_rho"]:.3f}, gid={m["gid"]})')
            report_lines.append(msg)

    # Summary interpretation
    report_lines.append('')
    report_lines.append(
        'Interpretation guide:')
    report_lines.append(
        '  THEORY OVER-PREDICTS -> measured erosion is slower than predicted.')
    report_lines.append(
        '    Possible causes: (1) alpha < 0.25 for this group; (2) phase mixing')
    report_lines.append(
        '    is decorrelating delta_rho and delta_v faster than l_perp captures;')
    report_lines.append(
        '    (3) slow-mode propagation introduces a spatial phase offset that')
    report_lines.append(
        '    reduces the effective flux below the isotropic slaving estimate.')
    report_lines.append(
        '  THEORY UNDER-PREDICTS -> measured erosion is faster than predicted.')
    report_lines.append(
        '    Possible causes: (1) alpha > 0.25; (2) additional compressive flux')
    report_lines.append(
        '    contributions not captured by the leading-order slaved estimate.')


# =============================================================================
#  FIG 4 — PREDICTED vs MEASURED WITH BEST-FIT ALPHA (per run)
# =============================================================================

def fig_bestfit_overlay(all_data):
    """
    Repeat the four-panel comparison of Fig 1, but with each dashed
    theoretical line drawn at its run-specific best-fit alpha rather than
    the fixed alpha=0.25.  The improvement in overlap relative to Fig 1
    quantifies how much of the closure error is attributable to a wrong
    prefactor versus a wrong functional form.
    """
    gids   = sorted(GROUP_META.keys())
    n_grps = len(gids)

    fig    = plt.figure(figsize=(6.5 * n_grps, 5.5))
    outer  = gridspec.GridSpec(1, n_grps, figure=fig,
                                wspace=0.38, left=0.07, right=0.97,
                                top=0.89, bottom=0.12)

    for col, gid in enumerate(gids):
        ax = fig.add_subplot(outer[col])
        runs_in_group = {
            tag: d for tag, d in all_data.items()
            if d['meta']['gid'] == gid and d['meta']['A_rho'] > 0}

        for tag, data in sorted(runs_in_group.items(),
                                key=lambda x: x[1]['meta']['A_rho']):
            t      = data['t']
            A_meas = data['A_rho_norm']
            color  = _color(data['meta']['A_rho'])
            ls_run = data['meta']['ls']
            lw_run = data['meta']['lw']
            A_rho  = data['meta']['A_rho']

            # Measured
            ax.semilogy(t, A_meas,
                        color=color, ls=ls_run, lw=lw_run,
                        label=rf'$A_\rho={A_rho:.3g}$, meas.')

            # Theory at alpha=0.25
            _, P_fixed, _ = build_prediction(data, ALPHA_THEORY)
            ax.semilogy(t, P_fixed,
                        color=color, ls=':', lw=0.9, alpha=0.50,
                        label=rf'$\alpha={ALPHA_THEORY}$ (theory)')

            # Theory at best-fit alpha
            alpha_bf, _ = calibrate_alpha(data)
            if np.isfinite(alpha_bf):
                _, P_bf, _ = build_prediction(data, alpha_bf)
                ax.semilogy(t, P_bf,
                            color=color, ls='--', lw=1.1, alpha=0.85,
                            label=rf'$\alpha={alpha_bf:.3f}$ (best-fit)')

        ax.axhline(0.5, color='grey', ls=':', lw=0.7, alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_xlabel(r'$t\,/\,t_A$')
        ax.set_ylabel(r'$A_\rho(t)\,/\,A_\rho(0)$')
        ax.set_title(GROUP_META[gid]['label'], fontsize=9.5, pad=4)
        ax.legend(fontsize=6.5, loc='lower left', ncol=1)
        _tick_both(ax)
        ax.grid(True, which='both', alpha=0.22)

    fig.suptitle(
        r'Best-fit $\alpha$ overlay.  Dotted: $\alpha=0.25$ (theory).  '
        r'Dashed: run-specific best-fit $\alpha$.',
        fontsize=10, y=0.97)
    savefig(fig, 'fig_bestfit_overlay.pdf')


# =============================================================================
#  TEXT REPORT
# =============================================================================

def write_report(report_lines):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'closure_diagnostics.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'\n  Text report -> {path}')


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print(__doc__)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = [
        'Fickian Closure Diagnostic Report',
        '=' * 70,
        f'Theoretical alpha: {ALPHA_THEORY}',
        f'Divergence threshold: {int(DIVERGENCE_THRESHOLD*100)}%',
        '=' * 70,
        '',
        'Per-run alpha calibration',
        '─' * 70,
    ]

    print('\n' + '='*65)
    print('Loading simulation data...')
    print('='*65)
    all_data = load_all_runs()

    if not all_data:
        print('ERROR: no data loaded.  Check OUTPUT_SPECS folder paths.')
        return

    n_gradient_runs = sum(1 for d in all_data.values()
                          if d['meta']['A_rho'] > 0)
    print(f'\n  Loaded {len(all_data)} runs total, '
          f'{n_gradient_runs} with A_rho > 0.')

    print('\n' + '='*65)
    print('Adversarial divergence check (alpha = 0.25)...')
    print('='*65)
    adversarial_divergence_check(all_data, report_lines)

    print('\n' + '='*65)
    print('Generating figures...')
    print('='*65)

    print('\nFig 1: Four-panel closure comparison (alpha=0.25)...')
    fig_closure_comparison(all_data)

    print('Fig 2: Best-fit alpha calibration scatter...')
    fig_alpha_calibration(all_data, report_lines)

    print('Fig 3: Alpha distribution by group (box/violin)...')
    fig_alpha_boxplot(all_data, report_lines)

    print('Fig 4: Best-fit alpha overlay comparison...')
    fig_bestfit_overlay(all_data)

    print('\nWriting text report...')
    write_report(report_lines)

    # Console summary of alpha by group
    print('\n' + '='*65)
    print('Alpha calibration summary:')
    print('  (Group 3 = elongated box L_x=5; Groups 0-2 = cubic)')
    print('-'*65)
    for line in report_lines:
        if 'Group' in line and ('mean' in line or 'ELONGATED' in line):
            print(f'  {line.strip()}')

    print('\n' + '='*65)
    print(f'All outputs written to:\n  {OUTPUT_DIR}')
    print('='*65)


if __name__ == '__main__':
    main()
