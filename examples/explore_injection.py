"""Explore an injection's SNR and likelihood without running a sampler.

Usage: python examples/explore_injection.py [config.yaml] (default
plot_config.yaml). Plots need the viz extra: pip install -e .[viz]
"""

# Keep per-eval BLAS/OMP single-threaded and neutralize lisatools' bare
# breakpoint(); must run before numpy is first imported to take effect.
from emridispatch.cli import set_env_guards

set_env_guards()

import os
import sys

from emridispatch.workbench import (
    load, measure, offset, prior_from_config, truth)


def main():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plot_config.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else default

    cfg, model = load(config_path)
    print(f"injection SNR = {model.optimal_snr:.4f}")
    print(f"add_noise = {getattr(model, 'add_noise', False)}, "
          f"seed = {getattr(model, 'noise_seed', None)}")

    at_truth = measure(model, truth(model), per_channel=True)
    print(f"at truth: optimal SNR = {at_truth.snr.optimal:.4f}, "
          f"detected = {at_truth.snr.detected:.4f}")
    print(f"          lnL = {at_truth.lnlike:.4f}, "
          f"overlap = {at_truth.overlap:.6f}")
    for name, chan in at_truth.snr.per_channel.items():
        print(f"          channel {name}: SNR = {chan.optimal:.4f}")

    shifted = measure(model, offset(model, p=+0.01))
    print(f"p + 0.01:  lnL = {shifted.lnlike:.4f}, "
          f"overlap = {shifted.overlap:.6f}")

    prior = prior_from_config(cfg, model=model)
    print(f"prior: {prior.ndim} parameters, mins {prior.mins[:3]} ...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from emridispatch.workbench_plots import (
            plot_char_strain, plot_snr_accumulation, plot_time_frequency)
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    figures = {
        "time_frequency": plot_time_frequency(model),
        "char_strain": plot_char_strain(model, show=("data", "template")),
        "snr_accumulation": plot_snr_accumulation(
            model, show=("data", "template")),
    }
    for name, (fig, _axes) in figures.items():
        out = f"explore_{name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
