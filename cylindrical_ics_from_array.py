"""
cylindrical_ic_rmhd_compact.py  (v3 — discrete-response weighted)
================================================================
z+/z- Alfvenic IC generator for Athena++ on a cylindrical annulus.
B0 = (C/r) phi_hat; parallel = phi; perpendicular = (r, z).

KEY DESIGN (this version):
 * SPECTRAL CONTROL VIA DISCRETE CT RESPONSE.  For each retained mode j we build
   a unit-amplitude A_phi_j, apply the ACTUAL discrete CT curl, measure the
   realized energy Q_j = sum |dB_j|^2 r dV, and set the coefficient
   A_j = sqrt(W_j / Q_j) where W_j is the target (Gaussian) energy weight.
   The initial spectrum therefore matches the target in the DISCRETE norm the
   solver sees, not in a continuum approximation (~5% different).
 * PREDICTED vs REALIZED chi_A, both reported.  Convention (NOTE: the review
   memo wrote the inverse):   chi_A = delta_v k_perp / (v_A k_par)  (>1 strong).
   - predicted: mode-wise distribution from the retained-mode list (median,
     energy-weighted geometric mean, 10-90 pct, m=0 fraction = P(chi=inf)).
   - realized:  from the initialized field itself via FFT in (phi, z) with
     k_r,eff = pi/L_r (labeled approximation).
 * ALL volume-weighted: m=0 energy fraction, perturbation means, <B0^2> for
   beta, use dV = r dr dphi dz.
 * Background check: the CT curl of A_z = -C ln r is verified against C/r_c
   (the discrete curl gives C at the LOG-MEAN radius of each cell; the printed
   deviation is the expected O(dr^2) discretization, not an error).
 * Anisotropy filter variables renamed per audit: kperp_eff_rmin(_sq) — an
   EFFECTIVE perpendicular wavenumber bound, not a literal Fourier k_r.
 * Compressibility uses the FULL 9-component cylindrical velocity-gradient
   tensor (incl. u_r/r, u_phi/r metric terms).
 * m=0 fraction is SUPPLEMENTARY; the primary RMHD criterion printed is the
   energy-weighted <k_par/k_perp>.
"""
import os
import numpy as np
from scipy.special import jv, yv
from scipy.optimize import brentq

import cylindrical_ic_gen_v3 as g
from cylindrical_ic_extra_v3 import _pack_and_write


# --------------------------------------------------------------------- Bessel
def _char(k, m, r0, r1):
    return yv(m, k*r0)*jv(m, k*r1) - jv(m, k*r0)*yv(m, k*r1)


def bessel_roots(m, r0, r1, n):
    ks = np.linspace(1e-6, (n + 4)*np.pi/(r1 - r0) + 6.0, 8000)
    f = _char(ks, m, r0, r1)
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(f[i]) and np.isfinite(f[i+1]) and f[i]*f[i+1] < 0:
            rt = brentq(_char, ks[i], ks[i+1], args=(m, r0, r1), xtol=1e-12)
            if not out or rt - out[-1] > 1e-6:
                out.append(rt)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"{len(out)}/{n} radial roots for m={m}")
    for rt in out:
        assert abs(_char(rt, m, r0, r1)) < 1e-8
    return np.array(out)


def R_mode(kap, m, r, r0):
    return yv(m, kap*r0)*jv(m, kap*r) - jv(m, kap*r0)*yv(m, kap*r)


# ----------------------------------------------------------------------- main
def create_ic(folder, h5name, n_X, X_min, X_max, meshblock,
              beta=1.0, dB_rms=0.15,
              kpeak=(1, 4), kwidth=4.0,
              m_max=6, nz_max=8, nr_max=6, eps_aniso=0.1,
              C_bg=1.0, rho0=1.0, amp_rho=0.0, sign_mode='plus', seed=42):
    EPS = 1e-30
    folder = folder if folder.endswith('/') else folder + '/'
    os.makedirs(folder, exist_ok=True)
    nr, nphi, nz = map(int, n_X)
    X_min = np.asarray(X_min, float); X_max = np.asarray(X_max, float)
    rmin, rmax = X_min[0], X_max[0]
    L_r, L_z = rmax - rmin, X_max[2] - X_min[2]
    r_mid = 0.5*(rmin + rmax)
    gamma = 5.0/3.0
    (r_c, phi_c, z_c), (r_face, phi_face, z_face), (dr, dphi, dz) = \
        g.cylindrical_mesh([nr, nphi, nz], X_min, X_max)
    rW = r_c[None, None, :]                       # volume weight r (dV = r dr dphi dz)

    B0sq_V = float(np.sum((C_bg/r_c)**2 * r_c)/np.sum(r_c))
    p0 = 0.5*beta*B0sq_V
    print(f"[ic] beta={beta} (vol-avg) -> p0={p0:.6g}; local beta in "
          f"[{2*p0/(C_bg/rmin)**2:.2f},{2*p0/(C_bg/rmax)**2:.2f}]")

    if abs(amp_rho) >= 1.0:
        raise ValueError("|amp_rho| must be < 1")
    rho_prof = rho0*(1.0 + amp_rho*np.sin(2*np.pi*(r_c - rmin)/L_r))
    assert rho_prof.min() > 0
    rho = np.broadcast_to(rho_prof[None, None, :], (nz, nphi, nr)).copy()

    # ---- pass 1: enumerate retained modes (audit item 1 naming) -------------
    kprl0, kprp0 = 2*np.pi*kpeak[0], 2*np.pi*kpeak[1]
    rng = np.random.default_rng(seed)
    modes = []
    skipped = 0
    for m in range(0, m_max + 1):
        roots = bessel_roots(m, rmin, rmax, nr_max)
        kpar_max = m/rmin                                  # largest k_par on annulus
        for n_z in range(0, nz_max + 1):
            k_z = 2*np.pi*n_z/L_z
            for kap in roots:
                # effective perpendicular bound at rmin (NOT a literal Fourier k_r)
                kperp_eff_rmin_sq = kap**2 + k_z**2 - (m/rmin)**2
                if kperp_eff_rmin_sq <= 0.0:
                    skipped += 1; continue
                kperp_eff_rmin = np.sqrt(kperp_eff_rmin_sq)
                if kpar_max > eps_aniso*kperp_eff_rmin:
                    skipped += 1; continue
                Kprl = m/r_mid
                Kprp = np.sqrt(max(kap**2 + k_z**2 - (m/r_mid)**2, EPS))
                env = np.exp(-((Kprl - kprl0)**2 + (Kprp - kprp0)**2)/kwidth**2)
                if env < 1e-7:
                    skipped += 1; continue
                modes.append(dict(m=m, n_z=n_z, kap=kap, k_z=k_z,
                                  th=rng.uniform(0, 2*np.pi),
                                  Kprl=Kprl, Kprp=Kprp, W=env**2))
    if not modes:
        raise RuntimeError("no modes survived; loosen eps_aniso/kpeak/kwidth")
    print(f"[ic] modes kept={len(modes)} skipped={skipped} (eps={eps_aniso} @ rmin)")

    # ---- pass 2: DISCRETE CT response weighting (audit item 2) --------------
    PH = phi_c[None, :, None]; ZF = z_face[:, None, None]
    A_r0 = np.zeros((nz + 1, nphi + 1, nr))
    A_z0 = np.zeros((nz, nphi + 1, nr + 1))
    def ct_curl(Aphi):
        return g.face_B_from_edge_A(A_r0, Aphi, A_z0, r_face, r_c, dr, dphi, dz)
    A_phi = np.zeros((nz + 1, nphi, nr + 1))
    for md in modes:
        Aj = R_mode(md['kap'], md['m'], r_face, rmin)[None, None, :] \
             * np.sin(md['m']*PH + md['k_z']*ZF + md['th'])
        bjr, bjp, bjz = ct_curl(Aj)
        cjr, cjp, cjz = g.cell_center_B_athena(bjr, bjp, bjz, r_face)
        Q = float(np.sum((cjr**2 + cjp**2 + cjz**2)*rW))     # discrete response
        md['A'] = np.sqrt(md['W']/(Q + EPS))
        A_phi += md['A']*Aj

    dBr_f, dBphi_f, dBz_f = ct_curl(A_phi)
    dBr_cc, dBphi_cc, dBz_cc = g.cell_center_B_athena(dBr_f, dBphi_f, dBz_f, r_face)
    m_int = ((r_c > rmin + 0.1*L_r) & (r_c < rmax - 0.1*L_r))
    M = np.broadcast_to(m_int[None, None, :], dBr_cc.shape)
    rms0 = np.sqrt(np.mean((dBr_cc**2 + dBphi_cc**2 + dBz_cc**2)[M]))
    A_phi *= dB_rms/(rms0 + EPS)
    dBr_f, dBphi_f, dBz_f = ct_curl(A_phi)

    # ---- background + EXPLICIT C/r verification (audit item 6) --------------
    A_z_bg = A_z0 + (-C_bg*np.log(r_face))[None, None, :]
    Br_bg, Bphi_bg, Bz_bg = g.face_B_from_edge_A(A_r0, np.zeros_like(A_phi), A_z_bg,
                                                 r_face, r_c, dr, dphi, dz)
    _, Bphi_bg_cc, _ = g.cell_center_B_athena(Br_bg, Bphi_bg, Bz_bg, r_face)
    Bphi_expected = C_bg/r_c
    bg_err = float(np.max(np.abs(Bphi_bg_cc[0, 0, :] - Bphi_expected)/Bphi_expected))
    print(f"[ic] background check: max|Bphi_CT - C/r|/(C/r) = {bg_err:.2e} "
          f"(expected O(dr^2) log-mean discretization)")

    Br_f = dBr_f + Br_bg; Bphi_f = dBphi_f + Bphi_bg; Bz_f = dBz_f + Bz_bg
    divB = g.discrete_divB_cylindrical(Br_f, Bphi_f, Bz_f, r_face, r_c, dr, dphi, dz)
    maxdiv = float(np.abs(divB).max())
    assert maxdiv < 1e-10, f"divB={maxdiv:.2e}"

    # ---- Elsasser velocity + write ------------------------------------------
    s = +1.0 if sign_mode == 'plus' else -1.0
    dBr_cc, dBphi_cc, dBz_cc = g.cell_center_B_athena(dBr_f, dBphi_f, dBz_f, r_face)
    sq = np.sqrt(rho + EPS)
    du_r, du_phi, du_z = s*dBr_cc/sq, s*dBphi_cc/sq, s*dBz_cc/sq
    path = _pack_and_write(folder, h5name, [nr, nphi, nz], meshblock, r_face,
                           rho, du_r, du_phi, du_z, Br_f, Bphi_f, Bz_f,
                           np.full((nz, nphi, nr), p0), gamma=gamma)

    # =========================== VALIDATION ===================================
    Vw = rW  # volume weight
    # (item 7) volume-weighted perturbation means
    meanB = [float(np.sum(f*Vw)/np.sum(np.ones_like(f)*Vw))
             for f in (dBr_cc, dBphi_cc, dBz_cc)]
    meanU = [float(np.sum(f*Vw)/np.sum(np.ones_like(f)*Vw))
             for f in (du_r, du_phi, du_z)]
    # sigma_c / purity
    db_r, db_p, db_z = dBr_cc/sq, dBphi_cc/sq, dBz_cc/sq
    dudb = (du_r*db_r + du_phi*db_p + du_z*db_z)[M].mean()
    du2 = (du_r**2 + du_phi**2 + du_z**2)[M].mean()
    db2 = (db_r**2 + db_p**2 + db_z**2)[M].mean()
    sigma_c = 2*dudb/(du2 + db2 + EPS)
    Em = 0.25*((du_r-db_r)**2 + (du_phi-db_p)**2 + (du_z-db_z)**2)[M].mean()
    Ep = 0.25*((du_r+db_r)**2 + (du_phi+db_p)**2 + (du_z+db_z)**2)[M].mean()
    realized = np.sqrt(np.mean((dBr_cc**2 + dBphi_cc**2 + dBz_cc**2)[M]))

    # (item 4) VOLUME-WEIGHTED m=0 energy fraction (supplementary; item 8)
    B2 = dBr_cc**2 + dBphi_cc**2 + dBz_cc**2
    B0m = (dBr_cc.mean(axis=1)**2 + dBphi_cc.mean(axis=1)**2
           + dBz_cc.mean(axis=1)**2)                       # (nz, nr)
    E_m0 = float(np.sum(B0m*r_c[None, :]))
    E_tot = float(np.sum(B2.mean(axis=1)*r_c[None, :]))
    frac_m0 = E_m0/(E_tot + EPS)

    # (item 5) compressibility with FULL 9-component cylindrical tensor
    gr = r_c[None, None, :]
    d = lambda f, x, ax: np.gradient(f, x, axis=ax)
    div_u = (d(gr*du_r, r_c, 2)/gr + d(du_phi, dphi, 1)/gr + d(du_z, dz, 0))
    G2 = (d(du_r, r_c, 2)**2 + ((d(du_r, dphi, 1)/gr) - du_phi/gr)**2 + d(du_r, dz, 0)**2
          + d(du_phi, r_c, 2)**2 + ((d(du_phi, dphi, 1)/gr) + du_r/gr)**2 + d(du_phi, dz, 0)**2
          + d(du_z, r_c, 2)**2 + (d(du_z, dphi, 1)/gr)**2 + d(du_z, dz, 0)**2)
    comp = np.sqrt((div_u**2)[M].mean())
    grad_frob = np.sqrt(G2[M].mean()) + EPS

    # (item 3) predicted chi_A distribution from mode list (weights = W, which
    # now EQUAL the realized per-mode energies by construction)
    v_A = (C_bg/r_mid)/np.sqrt(rho0); dv = dB_rms/np.sqrt(rho0)
    W = np.array([md['W'] for md in modes]); W /= W.sum() + EPS
    marr = np.array([md['m'] for md in modes])
    kpl = np.array([md['Kprl'] for md in modes])
    kpp = np.array([md['Kprp'] for md in modes])
    chi = np.where(kpl > 0, dv*kpp/(v_A*np.maximum(kpl, EPS)), np.inf)
    order = np.argsort(chi); cs = np.cumsum(W[order])
    pct = lambda q: chi[order][min(np.searchsorted(cs, q), len(chi)-1)]
    fin = np.isfinite(chi)
    logmean = (np.exp(np.sum(W[fin]*np.log(chi[fin]))/(W[fin].sum()+EPS))
               if fin.any() else np.inf)
    kratio_pred = float(np.sum(W*kpl/kpp))                # PRIMARY RMHD criterion

    # realized chi_A from the ACTUAL field (FFT, integer-rounded indices)
    m_idx = np.rint(np.fft.fftfreq(nphi, 1.0)*nphi).astype(int)
    nz_idx = np.rint(np.fft.fftfreq(nz, 1.0)*nz).astype(int)
    P = np.zeros((nz, nphi))
    for f in (db_r, db_p, db_z):
        F = np.fft.fft2(f[:, :, m_int], axes=(0, 1))
        P += (np.abs(F)**2).mean(axis=2)
    kr_eff = np.pi/(r_c[m_int][-1] - r_c[m_int][0])       # labeled approximation
    KPAR = np.abs(m_idx)[None, :]/r_mid
    KPERP = np.sqrt((2*np.pi*nz_idx[:, None]/L_z)**2 + kr_eff**2)
    kpar_r = float(np.sum(P*KPAR)/(np.sum(P) + EPS))
    kperp_r = float(np.sum(P*KPERP)/(np.sum(P) + EPS))
    dv_real = np.sqrt(du2)
    chi_real = dv_real*kperp_r/(v_A*(kpar_r + EPS))
    # realized parallel peak (where does the m-spectrum actually peak?)
    Pm = np.array([P[:, m_idx == k].sum() for k in range(nphi//2)])
    m_peak = int(np.argmax(Pm))

    print(f"[ic] wrote {path}")
    print(f"[ic] divB={maxdiv:.2e}  sigma_c={sigma_c:+.4f}  E-/(E+ + E-)={Em/(Ep+Em+EPS):.2e}")
    print(f"[ic] realized dB_rms={realized:.4f} (target {dB_rms}); "
          f"per-mode energies = target W_j via DISCRETE CT response")
    print(f"[ic] vol-weighted means: <dB>=({meanB[0]:+.1e},{meanB[1]:+.1e},{meanB[2]:+.1e})"
          f"  <du>=({meanU[0]:+.1e},{meanU[1]:+.1e},{meanU[2]:+.1e})")
    print(f"[ic] PRIMARY RMHD criterion: energy-weighted <k_par/k_perp> = {kratio_pred:.4f}"
          f"  (m=0 fraction, r-weighted, supplementary: {frac_m0:.3f})")
    print(f"[ic] compressibility: rms(div u)/rms(|curl-free-normed grad u|) "
          f"= {comp/grad_frob:.3e}  [full 9-component cyl tensor]")
    print(f"[ic] chi_A convention: chi = dv*k_perp/(v_A*k_par)  (>1 strong)")
    print(f"[ic]   PREDICTED (mode-wise, W-weighted): median={pct(0.5):.2f} "
          f"mean(log)={logmean:.2f} 10-90=[{pct(0.1):.2f},"
          f"{pct(0.9) if np.isfinite(pct(0.9)) else np.inf}] m0-frac={float(W[marr==0].sum()):.3f}")
    print(f"[ic]   REALIZED  (field FFT, kr_eff=pi/Lr): chi={chi_real:.2f} "
          f"(k_par_eff={kpar_r:.2f}, k_perp_eff={kperp_r:.1f}, dv_rms={dv_real:.3f})")
    print(f"[ic] realized PARALLEL peak at m={m_peak} "
          f"(target kpeak_par={kpeak[0]} -> k_par={kprl0:.1f} -> m~{kprl0*r_mid:.0f}; "
          f"if filter-capped below this, raise eps_aniso)")
    return path


if __name__ == '__main__':
    create_ic('/tmp/rmhd_v3/', 'ics.h5', n_X=[64, 64, 32],
              X_min=[0.8, 0, 0], X_max=[1.2, 2*np.pi, 1.0], meshblock=[32, 32, 16],
              dB_rms=0.15, kpeak=(1, 4), kwidth=4.0,
              m_max=8, nz_max=8, nr_max=6, eps_aniso=0.1)
