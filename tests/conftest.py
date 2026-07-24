import json
import textwrap

import numpy as np
import pytest

from emridispatch.config import load_config
from emridispatch.parameters import NDIM, PARAM_NAMES

MINIMAL_TOY_CONFIG = """
injection:
  mass_1: 1.0e+6
  mass_2: 10.0
  a: 0.0
  p: 10.0
  e: 0.1
  x: 1.0
  q_k: 1.0
  phi_k: 1.5707963267948966
  q_s: 1.0
  phi_s: 1.5707963267948966
  luminosity_distance: 1.0
  phi_phi: 1.5707963267948966
  phi_theta: 1.5707963267948966
  phi_r: 1.5707963267948966

data:
  response: toy
  duration: 0.3
  delta_t: 10.0
  inj_snr: 30.0
  channels: [A, E]

sampler:
  nsamples: 200
  impulse:
    threads: 1
    cov_update: 50
    save_freq: 50
    ladder:
      max_temp: 100.0
      t_split: 10.0
      ntemps_low: 3
      ntemps_high: 2
    mode_jump:
      method: none
      weight: 25.0

prior:
  fisher: none
  angle_sigma: 0.05
  fisher_use_gpu: false

reparam:
  mode: auto
  idx: [0, 1, 2, 3, 4, 5]

run:
  seed: 42
  outdir: {outdir}
"""


@pytest.fixture
def toy_cfg(tmp_path):
    """Loaded config for the toy response model with heuristic Fisher."""
    outdir = tmp_path / "chains"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(textwrap.dedent(
        MINIMAL_TOY_CONFIG.format(outdir=str(outdir))))
    return load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# Fake raw run dirs (shared by test_results / test_diagnostics / test_multichain)
# ---------------------------------------------------------------------------

NSTEPS = 200
TEMPS = [1.0, 2.5]
TRUTH = np.linspace(0.1, 1.2, NDIM)


def make_run_dir(tmp_path, with_truth=True, with_summary=True, with_spec=False):
    """Fake impulse run dir: chain_N.txt per rung + shared sidecars."""
    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for rung, temp in enumerate(TEMPS):
        cols = np.empty((NSTEPS, NDIM + 4))
        cols[:, :NDIM] = TRUTH + 0.05 * rng.standard_normal((NSTEPS, NDIM))
        cols[:, NDIM] = -0.5 * rng.random(NSTEPS)      # lnlike
        cols[:, NDIM + 1] = cols[:, NDIM] - 1.0        # lnprob
        cols[:, NDIM + 2] = 1.0                        # accepted
        cols[:, NDIM + 3] = temp
        np.savetxt(tmp_path / f"chain_{rung}.txt", cols)
    if with_truth:
        truth = {
            "param_names": PARAM_NAMES,
            "sampling_vector": TRUTH.tolist(),
            "injection": {"mass_1": 1e6, "mass_2": 10.0},
        }
        (tmp_path / "injection_truth.json").write_text(json.dumps(truth))
    if with_summary:
        summary = {"config": {"backend": "impulse", "ntemps": len(TEMPS)}}
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    if with_spec:
        from emridispatch.priors import JointPrior, Uniform

        prior = JointPrior(
            [Uniform(-5.0, 5.0) for _ in range(NDIM)], names=PARAM_NAMES)
        (tmp_path / "prior_spec.json").write_text(json.dumps(prior.spec()))
    (tmp_path / "config.yaml").write_text(
        "run:\n  seed: 42\nsampler:\n  backend: impulse\n")
    (tmp_path / "run.log").write_text("versions: emridispatch=0.1.0\n")
    return tmp_path


ERYN_NT, ERYN_NW, ERYN_IT = 2, 3, 5


def make_eryn_run_dir(tmp_path):
    """Synthetic eryn_chain.h5 mirroring eryn's HDFBackend layout, with an
    over-allocated tail beyond `iteration` and a beta=0 top rung."""
    import h5py

    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    alloc = ERYN_IT + 4                      # datasets outlive the iteration
    chain = np.zeros((alloc, ERYN_NT, ERYN_NW, 1, NDIM))
    chain[:ERYN_IT] = rng.standard_normal((ERYN_IT, ERYN_NT, ERYN_NW, 1, NDIM))
    # Known values in param 0 to pin down the step-major flattening order.
    chain[:ERYN_IT, :, :, 0, 0] = (np.arange(ERYN_IT)[:, None, None]
                                   + 0.1 * np.arange(ERYN_NW)[None, None, :])
    log_like = np.zeros((alloc, ERYN_NT, ERYN_NW))
    log_like[:ERYN_IT] = -rng.random((ERYN_IT, ERYN_NT, ERYN_NW))
    log_prior = np.zeros((alloc, ERYN_NT, ERYN_NW))
    log_prior[:ERYN_IT] = -1.0
    betas = np.zeros((alloc, ERYN_NT))
    betas[:ERYN_IT] = [1.0, 0.0]             # cold rung + T=inf top rung
    accepted = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])

    with h5py.File(tmp_path / "eryn_chain.h5", "w") as f:
        g = f.create_group("mcmc")
        g.attrs["iteration"] = ERYN_IT
        g.attrs["ntemps"] = ERYN_NT
        g.attrs["nwalkers"] = ERYN_NW
        g.create_dataset("chain/model_0", data=chain)
        g.create_dataset("log_like", data=log_like)
        g.create_dataset("log_prior", data=log_prior)
        g.create_dataset("betas", data=betas)
        g.create_dataset("accepted", data=accepted)
        g.create_dataset("swaps_accepted", data=np.array([2.0]))

    truth = {"param_names": PARAM_NAMES, "sampling_vector": TRUTH.tolist(),
             "injection": {"mass_1": 1e6, "mass_2": 10.0}}
    (tmp_path / "injection_truth.json").write_text(json.dumps(truth))
    summary = {"config": {"backend": "eryn", "ntemps": ERYN_NT,
                          "nwalkers": ERYN_NW}}
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    (tmp_path / "config.yaml").write_text(
        "run:\n  seed: 42\nsampler:\n  backend: eryn\n")
    return tmp_path
