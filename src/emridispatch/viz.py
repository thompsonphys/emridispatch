"""Posterior visualization from the common results.h5 format.

Sampler-agnostic by construction: this module only reads results files written
by emridispatch-postprocess (see emridispatch.results), never raw backend output.

    emridispatch-plot OUTDIR                # corner + 1D marginals of the posterior
    emridispatch-plot OUTDIR --all-temps    # marginals overlaid across the ladder
    emridispatch-plot results.h5 --temps 0 3 5 --burn 50

Default coordinates are physical (whitening inverted); --sampling plots the raw
sampling coordinates instead -- note that for whitened runs the intrinsic block
is then in rotated/scaled coordinates, only labeled by its physical names.

Requires matplotlib + corner:

    pip install emridispatch[viz]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from emridispatch.parameters import PARAM_LABELS, PARAM_NAMES
from emridispatch.results import DEFAULT_NAME, Results

CORNER_NAME = "corner.png"
MARGINALS_NAME = "marginals.png"


def _require_plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import corner
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib + corner; install with "
            "`pip install emridispatch[viz]`") from exc
    return plt, corner


def resolve_results_path(arg):
    """Accept a results .h5 file or a directory containing results.h5."""
    if os.path.isdir(arg):
        path = os.path.join(arg, DEFAULT_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no {DEFAULT_NAME} in {arg}; run `emridispatch-postprocess "
                f"{arg}` first")
        return path
    if not os.path.exists(arg):
        raise FileNotFoundError(f"{arg}: no such file or directory")
    return arg


def _labels(results):
    """LaTeX labels when the file's parameter set matches ours, else raw names."""
    if list(results.param_names) == list(PARAM_NAMES):
        return list(PARAM_LABELS)
    return list(results.param_names)


def plot_corner(samples, labels, truth=None, out_path="corner.png"):
    """Corner plot of one rung's samples with 16/50/84% titles."""
    plt, corner = _require_plotting()
    fig = corner.corner(
        samples, labels=labels, truths=truth, show_titles=True,
        quantiles=(0.16, 0.5, 0.84), title_fmt=".3g")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_marginals(rung_data, labels, truth=None, out_path="marginals.png"):
    """Grid of 1-D marginals; one histogram per rung, colored by temperature.

    rung_data: list of (temperature, samples[N, ndim]) tuples, cold first.
    """
    plt, _corner = _require_plotting()
    ndim = rung_data[0][1].shape[1]
    ncols = 4
    nrows = int(np.ceil(ndim / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("viridis")
    nrungs = len(rung_data)
    for j in range(ndim):
        ax = axes[j]
        for k, (temp, samples) in enumerate(rung_data):
            color = cmap(k / max(nrungs - 1, 1)) if nrungs > 1 else "C0"
            ax.hist(samples[:, j], bins=50, density=True, histtype="step",
                    color=color, label=f"$T$ = {temp:g}")
        if truth is not None:
            ax.axvline(truth[j], color="k", ls="--", lw=1)
        ax.set_xlabel(labels[j])
        ax.set_yticks([])
    for ax in axes[ndim:]:
        ax.set_visible(False)
    fig.tight_layout()
    if nrungs > 1:
        # Outside the axes grid on the right; bbox_inches="tight" includes it.
        handles, lab = axes[0].get_legend_handles_labels()
        fig.legend(handles, lab, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=6, frameon=False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_plots(results, temps=(0,), all_temps=False, burn=0, physical=True,
               thin=1, outdir="."):
    """Produce corner.png (coldest requested rung) + marginals.png (all
    requested rungs overlaid). Returns the written paths."""
    if all_temps:
        rungs = list(range(results.ntemps))
    else:
        rungs = sorted(set(int(t) for t in temps))
    bad = [r for r in rungs if not 0 <= r < results.ntemps]
    if bad:
        raise ValueError(
            f"requested rung(s) {bad} out of range; this run has "
            f"{results.ntemps} rungs (0..{results.ntemps - 1})")

    labels = _labels(results)
    truth = results.truth_physical if physical else results.truth_sampling
    rung_data = [
        (float(results.temperatures[r]),
         results.rung(r, physical=physical, burn=burn, thin=thin))
        for r in rungs
    ]

    os.makedirs(outdir, exist_ok=True)
    written = []
    cold_idx = rungs.index(min(rungs))
    written.append(plot_corner(
        rung_data[cold_idx][1], labels, truth=truth,
        out_path=os.path.join(outdir, CORNER_NAME)))
    written.append(plot_marginals(
        rung_data, labels, truth=truth,
        out_path=os.path.join(outdir, MARGINALS_NAME)))
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Corner + 1D posterior plots from a results.h5 file "
                    "(produce one with emridispatch-postprocess).")
    ap.add_argument("results",
                    help="results.h5 path, or a directory containing one")
    ap.add_argument("--burn", type=int, default=0,
                    help="draws to drop from the front of each rung")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--temps", type=int, nargs="+", default=[0],
                       help="temperature rungs to overlay in the marginals "
                            "(default: 0, the cold chain / posterior)")
    group.add_argument("--all-temps", action="store_true",
                       help="overlay the full temperature ladder")
    ap.add_argument("--sampling", action="store_true",
                    help="plot raw sampling coordinates instead of physical")
    ap.add_argument("--thin", type=int, default=1,
                    help="cosmetic thinning for plot rendering only")
    ap.add_argument("--outdir", default=None,
                    help="where to write the pngs (default: beside the "
                         "results file)")
    args = ap.parse_args()

    try:
        path = resolve_results_path(args.results)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    results = Results.load(path)
    outdir = args.outdir or os.path.dirname(os.path.abspath(path))

    if results.truth_physical is None:
        print("note: no injection truth stored; plotting without truth overlay")

    written = make_plots(
        results, temps=args.temps, all_temps=args.all_temps, burn=args.burn,
        physical=not args.sampling, thin=args.thin, outdir=outdir)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
