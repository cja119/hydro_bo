"""Bootstrap calibration QQ plots and Shapiro-Wilk statistics."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

LH2_PATH = 'LH2.json'
NH3_PATH = 'NH3.json'
OUT_STEM = 'qq_calibration'

N = 32
B = 10_000
LAM = 0.5
SIGMA0_SQ = 10.0
SEED_BOOT = 2026
SEED_SW = 0
N_SW = 500


def load_dataset(path):
    """Return feasible scores and the binary feasibility series."""
    raw = json.load(open(path))['worker_scores']
    scores = np.array([s for s in raw if s is not None], dtype=float)
    feas = np.array([0 if s is None else 1 for s in raw], dtype=int)
    return scores, feas


def pg_vi(c, n, sigma0_sq, max_iter=200, tol=1e-14):
    """Single-point PG-VI fixed point, returning (mean, variance)."""
    mu, sigma_sq = 0.0, sigma0_sq
    for _ in range(max_iter):
        xi = np.sqrt(mu**2 + sigma_sq)
        e_nu = n / 4.0 if xi < 1e-8 else (n / (2 * xi)) * np.tanh(xi / 2)
        new_sigma_sq = 1.0 / (e_nu + 1.0 / sigma0_sq)
        new_mu = new_sigma_sq * (c - n / 2)
        if abs(new_mu - mu) < tol and abs(new_sigma_sq - sigma_sq) < tol:
            break
        mu, sigma_sq = new_mu, new_sigma_sq
    return mu, sigma_sq


def z_mean(bm, m_bar, var_hat):
    """Sample mean against its CLT reference."""
    return (bm - m_bar) / np.sqrt(var_hat / N)


def z_log_var(bs2, var_hat):
    """Log sample variance against the chi-squared asymptotic."""
    mu_lv = np.log(var_hat) - 1.0 / (N - 1)
    return (np.log(bs2) - mu_lv) / np.sqrt(2.0 / (N - 1))


def z_rho(bm, bs2, m_bar, var_hat):
    """Risk-adjusted objective against its moment-matched reference."""
    m_sig, k_sig = np.log(var_hat), 2.0 / (N - 1)
    m_rho = m_bar - LAM * np.exp(0.5 * m_sig + 0.125 * k_sig)
    s2_rho = var_hat / N + LAM**2 * (np.exp(0.25 * k_sig) - 1) \
        * np.exp(m_sig + 0.25 * k_sig)
    return (bm - LAM * np.sqrt(bs2) - m_rho) / np.sqrt(s2_rho)


def z_feasibility(feas, rng):
    """PG-VI latent draws against the delta-method plus PG reference."""
    p = feas.mean()
    f_star = np.log(p / (1 - p))
    counts = feas[rng.integers(0, len(feas), size=(B, N))].sum(axis=1)
    mus, sigs = np.empty(B), np.empty(B)
    for i in range(B):
        mus[i], sigs[i] = pg_vi(counts[i], N, SIGMA0_SQ)
    f_draws = mus + np.sqrt(sigs) * rng.standard_normal(B)
    s2_f = 1.0 / (N * p * (1 - p)) \
        + 1.0 / (N * (2 * p - 1) / (2 * f_star) + 1.0 / SIGMA0_SQ)
    return (f_draws - f_star) / np.sqrt(s2_f)


def z_series(path):
    """The four standardised bootstrap series for one dataset."""
    scores, feas = load_dataset(path)
    m_bar, var_hat = scores.mean(), scores.var(ddof=1)
    rng = np.random.default_rng(SEED_BOOT)
    sp = scores[rng.integers(0, len(scores), size=(B, N))]
    bm, bs2 = sp.mean(axis=1), sp.var(axis=1, ddof=1)
    return {r'$\bar{X}$': np.sort(z_mean(bm, m_bar, var_hat)),
            r'$\log s^2$': np.sort(z_log_var(bs2, var_hat)),
            r'$\rho$': np.sort(z_rho(bm, bs2, m_bar, var_hat)),
            r'$f$': np.sort(z_feasibility(feas, rng))}


def apply_style():
    """Old-school PSE rcParams; returns the single-column width."""
    base = 8
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': base, 'axes.labelsize': base,
        'xtick.labelsize': base - 1, 'ytick.labelsize': base - 1,
        'axes.titlesize': base, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })
    return 3.46


def qq_panel(ax, theo, eq):
    """One square QQ panel with symmetric range and matched ticks."""
    lim = max(abs(theo.min()), abs(theo.max()),
              abs(eq.min()), abs(eq.max())) * 1.03
    ax.plot(theo, eq, '.', color='black', markersize=1.2, rasterized=True)
    ax.plot((-lim, lim), (-lim, lim), color='black',
            linestyle='--', linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, symmetric=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, symmetric=True))
    ax.set_xlabel('Theoretical')


def make_figure(results):
    """Four-row stack, LH2 left and NH3 right."""
    metrics = list(next(iter(results.values())).keys())
    probs = (np.arange(1, B + 1) - 0.5) / B
    theo = stats.norm.ppf(probs)
    fig = plt.figure(figsize=(apply_style(), 7.2))
    box_w, box_h = 0.34, 0.155
    for y, metric in zip([0.82, 0.58, 0.34, 0.10], metrics):
        axes = [fig.add_axes([0.13, y, box_w, box_h]),
                fig.add_axes([0.62, y, box_w, box_h])]
        for ax, res in zip(axes, results.values()):
            qq_panel(ax, theo, res[metric])
        axes[0].set_ylabel('Observed')
        fig.text(0.5, y + box_h + 0.008, metric,
                 ha='center', va='bottom', fontsize=9)
    plt.savefig(f'{OUT_STEM}.png', dpi=600, bbox_inches='tight')
    print(f'Saved: {OUT_STEM}.png\n')


def report_shapiro(results):
    """Shapiro-Wilk on a fixed-seed sub-sample of each series."""
    rng = np.random.default_rng(SEED_SW)
    for lbl, res in results.items():
        print(lbl)
        for metric, z in res.items():
            idx = rng.choice(len(z), size=N_SW, replace=False)
            w, p_val = stats.shapiro(z[idx])
            name = {r'$\bar{X}$': 'Xbar', r'$\log s^2$': 'log s2',
                    r'$\rho$': 'rho', r'$f$': 'f'}[metric]
            print(f'  {name:8s}: mean(z) = {z.mean():+.3f}, '
                  f'sd(z) = {z.std(ddof=1):.3f}, '
                  f'W = {w:.4f}, p = {p_val:.3e}')
        print()


if __name__ == '__main__':
    results = {'LH2': z_series(LH2_PATH), 'NH3': z_series(NH3_PATH)}
    make_figure(results)
    report_shapiro(results)
