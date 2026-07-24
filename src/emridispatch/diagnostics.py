"""Convergence / mixing diagnostics for the EMRI cold chain(s).

Reads the cold rung of one or more run directories through the common results
layer (results.h5 when present, else an in-memory convert of the raw backend
output -- diagnostics never parse backend-specific chain files) and reports
the standard MCMC health metrics, using arviz (rank-normalized split-R-hat and
bulk/tail ESS, Vehtari+2021) with an emcee integrated-autocorrelation-time
cross-check. Requires the diagnostics extra: `pip install emridispatch[diagnostics]`.

Only the cold rung is a posterior sample, as the hot rungs are tempered and are
not drawn from the target. Independent cold chains therefore come from separate
runs (different seeds / outdirs), not from the other temperature files.

Ensemble backends (eryn): the stored cold chain interleaves nwalkers walkers
step-major. Walkers are coupled by the ensemble move, so they are NOT treated
as independent chains for R-hat -- R-hat always compares whole runs (seeds).
Autocorrelation-based stats (tau, ESS, Geweke) instead need genuine time
series, so the flattened chain is un-interleaved back into per-walker series
(emcee's canonical estimator; walker coupling makes the total ESS mildly
optimistic). `burn` means time steps per chain for every backend (for eryn one
step = nwalkers stored rows).

Coordinates: by default diagnostics run in the sampling coordinates the sampler
actually explored (the right space to judge mixing). With reparam mode "auto" the
intrinsic block (cols 0..5) is whitened, so those columns are u-coords; pass
--physical to invert them back to physical params (masses in log) for the summary.

Usage
-----
    # single run: tau / ESS / Geweke burn-in (R-hat needs >= 2 chains)
    emridispatch-diagnostics chains_emri

    # several independent seeds -> real R-hat across cold chains
    emridispatch-diagnostics chains_seed1 chains_seed2 chains_seed3

    # apply a burn-in, or let Geweke suggest one
    emridispatch-diagnostics chains_emri --burn 2000
    emridispatch-diagnostics chains_emri --sweep
"""

import argparse
import logging

import numpy as np

from emridispatch.parameters import PARAM_NAMES
from emridispatch.results import load_or_convert

logger = logging.getLogger(__name__)

# emcee.integrated_time(quiet=True) still logs a "chain shorter than 50*tau"
# warning per call; on a short run that is dozens of lines of noise. We already
# surface short-chain caveats via ess/tau, so quiet the emcee logger.
logging.getLogger("emcee.autocorr").setLevel(logging.ERROR)

def _chain_data(res, rung, physical):
    """(samples, lnlike) of one rung from a Results, honoring `physical`."""
    data = res.physical if physical and res.physical is not None else res.samples
    return data[rung], res.lnlike[rung]


def load_chain(run_dir, rung=0, physical=False):
    """Return (samples[N, NDIM], lnlike[N]) for one temperature rung of a run dir.

    Loads through the common results layer (results.h5 first, else convert);
    rung 0 is the cold chain. physical=True selects the reparam-inverted
    coordinates when the run stored a whitening transform (a temperature-
    independent coordinate map, so valid for hot rungs too).
    """
    res = load_or_convert(run_dir)
    samples, lnlike = _chain_data(res, int(rung), physical)
    return samples.copy(), lnlike.copy()


def load_cold_chain(run_dir, physical=False):
    """Return (samples[N, NDIM], lnlike[N]) for the cold rung of one run dir."""
    return load_chain(run_dir, rung=0, physical=physical)


def _walker_series(res, samples, lnlike):
    """Un-interleave one run's cold chain into per-walker time series.

    Returns (samples[nsteps, nw, ndim], lnlike[nsteps, nw]). Single-chain
    backends (impulse) get nw=1. eryn's step-major flattening is inverted
    exactly via config nwalkers; if that is missing or inconsistent, fall back
    to nw=1 with a warning (autocorrelation stats then see interleaved rows).
    """
    nw = 1
    if res.backend == "eryn":
        nw = int(res.config.get("nwalkers") or 0)
        if nw <= 0 or len(samples) % nw:
            logger.warning(
                "eryn results without a usable config nwalkers (%r); treating "
                "the flattened chain as one series -- tau/ESS will be distorted "
                "by walker interleaving", res.config.get("nwalkers"))
            nw = 1
    nsteps = len(samples) // nw
    return (samples[:nsteps * nw].reshape(nsteps, nw, -1),
            lnlike[:nsteps * nw].reshape(nsteps, nw))


def stack_chains(run_dirs, burn=0, physical=False):
    """Load all cold rungs, drop `burn` time steps, truncate to common lengths.

    Returns (post, series), both name -> array dicts for arviz.from_dict:
      post    (nruns, ndraw)    one flattened chain per run dir -- the R-hat
                                view (walkers within a run are coupled, so
                                only whole runs count as independent chains)
      series  (nchains, nsteps) per-walker time series pooled over runs -- the
                                autocorrelation view for tau/ESS/Geweke
    For single-chain backends the two views hold identical data.
    """
    cols, lnls, wcols, wlnls = [], [], [], []
    for d in run_dirs:
        res = load_or_convert(d)
        s, ll = _walker_series(res, *_chain_data(res, 0, physical))
        s, ll = s[burn:], ll[burn:]
        cols.append(s.reshape(-1, s.shape[-1]))   # step-major, like storage
        lnls.append(ll.reshape(-1))
        wcols.extend(s[:, w, :] for w in range(s.shape[1]))
        wlnls.extend(ll[:, w] for w in range(ll.shape[1]))

    n = min(len(c) for c in cols)
    if n <= 1:
        raise ValueError(f"after burn={burn} only {n} draws remain; reduce --burn")
    cols = [c[:n] for c in cols]
    lnls = [l[:n] for l in lnls]
    post = {name: np.stack([c[:, j] for c in cols]) for j, name in enumerate(PARAM_NAMES)}
    post["lnlike"] = np.stack(lnls)

    m = min(len(c) for c in wcols)
    series = {name: np.stack([c[:m, j] for c in wcols])
              for j, name in enumerate(PARAM_NAMES)}
    series["lnlike"] = np.stack([l[:m] for l in wlnls])
    return post, series


def emcee_tau(series):
    """Per-parameter integrated autocorr time (emcee), pooled over chains.

    `series` must hold genuine time series (per-walker for ensemble backends).
    Returns a dict name -> tau (NaN if the chain is too short to estimate).
    """
    from emcee.autocorr import integrated_time

    taus = {}
    for name, x in series.items():
        # x is (nchain, ndraw); emcee wants (ndraw, nwalkers) and averages walkers.
        try:
            taus[name] = float(integrated_time(x.T, quiet=True)[0])
        except Exception:
            taus[name] = np.nan
    return taus


def geweke_z(x1d, first=0.1, last=0.5):
    """Geweke z for one 1-D chain: (mean_a - mean_b) / sqrt(se_a^2 + se_b^2).

    Standard errors use the emcee autocorr time of each window, so the comparison
    accounts for within-window correlation. Returns NaN if a window is degenerate.
    """
    from emcee.autocorr import integrated_time

    n = len(x1d)
    a = x1d[: int(first * n)]
    b = x1d[int((1 - last) * n):]
    if len(a) < 5 or len(b) < 5:
        return np.nan

    def se(w):
        v = np.var(w, ddof=1)
        if v == 0:
            return 0.0
        try:
            tau = max(1.0, float(integrated_time(w, quiet=True)[0]))
        except Exception:
            tau = 1.0
        return np.sqrt(v * tau / len(w))

    sea, seb = se(a), se(b)
    denom = np.hypot(sea, seb)
    if denom == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / denom


def max_abs_geweke(series):
    """Max |Geweke z| over every parameter and time-series chain."""
    zmax = 0.0
    worst = None
    for name, x in series.items():
        for c in range(x.shape[0]):
            z = geweke_z(x[c])
            if np.isfinite(z) and abs(z) > zmax:
                zmax, worst = abs(z), (name, c, z)
    return zmax, worst


def report(run_dirs, burn=0, physical=False):
    import arviz as az

    post, series = stack_chains(run_dirs, burn=burn, physical=physical)
    nrun, ndraw = post["lnlike"].shape
    nseries, nsteps = series["lnlike"].shape
    # ESS/tau/Geweke from the per-walker time series (honest autocorrelation);
    # R-hat only across independent runs -- coupled walkers must not count.
    idata = az.from_dict({"posterior": series})

    walkers = f" ({nseries} walker series x {nsteps} steps)" if nseries > nrun else ""
    print(f"\ncold runs: {nrun}   draws/run (post-burn {burn} steps): {ndraw}"
          f"{walkers}   coords: {'physical' if physical else 'sampling'}")
    if nrun < 2:
        print("R-hat needs >= 2 independent cold runs (separate seeds/outdirs); "
              "showing ESS/tau only.")

    summ = az.summary(idata)
    taus = emcee_tau(series)
    # Append emcee tau + arviz-implied tau (= total draws / ess_bulk) per row.
    total = nseries * nsteps
    summ = summ.copy()
    if nrun >= 2:
        rhat = az.rhat(az.from_dict({"posterior": post}))
        summ["r_hat"] = [round(float(rhat[i]), 3) for i in summ.index]
    else:
        summ["r_hat"] = np.nan
    summ["tau_emcee"] = [round(taus.get(i, np.nan), 1) for i in summ.index]
    summ["tau_arviz"] = [round(total / e, 1) if e and e > 0 else np.nan
                         for e in summ["ess_bulk"]]
    import pandas as pd  # arviz dep; safe

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n", summ[["mean", "sd", "r_hat", "ess_bulk", "ess_tail",
                          "tau_emcee", "tau_arviz", "mcse_mean"]])

    zmax, worst = max_abs_geweke(series)
    tag = "OK (<2)" if zmax < 2 else "STATIONARITY SUSPECT (>=2)"
    wtxt = f"  worst: {worst[0]} chain {worst[1]} z={worst[2]:.2f}" if worst else ""
    print(f"\nGeweke max|z| = {zmax:.2f}  {tag}{wtxt}")

    # Headline numbers.
    ess_min = float(np.nanmin(summ["ess_bulk"].values))
    if nrun >= 2:
        rhat_max = float(np.nanmax(summ["r_hat"].values))
        print(f"\nheadline:  max R-hat = {rhat_max:.3f}"
              + ("  (converged, <1.01)" if rhat_max < 1.01 else "  (NOT converged, >=1.01)"))
    else:
        print("\nheadline:  R-hat = n/a (need >= 2 cold runs)")
    print(f"           min ess_bulk = {ess_min:.0f}   "
          f"(total draws {total}; quote MC error via ESS, do NOT thin)")
    return summ


def sweep_burn(run_dirs, physical=False, fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """Try a grid of burn-in fractions; recommend the smallest that passes Geweke.

    Burn is in time steps per chain (see stack_chains); ESS/Geweke run on the
    per-walker time series.
    """
    _, series0 = stack_chains(run_dirs, burn=0, physical=physical)
    n = series0["lnlike"].shape[1]
    print(f"\nburn-in sweep (steps/chain = {n}):")
    print(f"  {'burn':>7} {'frac':>5}  {'max|z|':>7}  {'min_ess':>8}  verdict")
    rec = None
    for f in fracs:
        b = int(f * n)
        _, series = stack_chains(run_dirs, burn=b, physical=physical)
        import arviz as az

        ess_ds = az.ess(az.from_dict({"posterior": series}), method="bulk")
        ess_vals = [float(ess_ds[name]) for name in series]
        ess_min = float(np.nanmin(ess_vals))
        zmax, _ = max_abs_geweke(series)
        ok = zmax < 2
        if ok and rec is None:
            rec = b
        print(f"  {b:>7} {f:>5.2f}  {zmax:>7.2f}  {ess_min:>8.0f}  "
              f"{'pass' if ok else 'fail'}")
    if rec is not None:
        print(f"\nrecommend burn-in = {rec} steps (smallest passing Geweke |z|<2)")
    else:
        print("\nno tested burn-in passed |z|<2 -> chain not yet stationary; "
              "run longer or improve mixing (ladder / mode jumps)")
    return rec


def main():
    p = argparse.ArgumentParser(description="EMRI cold-chain convergence diagnostics")
    p.add_argument("run_dirs", nargs="+",
                   help="one or more run output dirs (any backend; results.h5 "
                        "read when present, raw output converted otherwise)")
    p.add_argument("--burn", type=int, default=0,
                   help="drop this many time steps from each chain front")
    p.add_argument("--sweep", action="store_true", help="scan burn-in fractions and recommend one")
    p.add_argument("--physical", action="store_true", help="invert reparam whitening to physical coords")
    args = p.parse_args()

    if args.sweep:
        sweep_burn(args.run_dirs, physical=args.physical)
    report(args.run_dirs, burn=args.burn, physical=args.physical)


if __name__ == "__main__":
    main()
