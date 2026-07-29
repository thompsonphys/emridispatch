import json
import textwrap

import numpy as np
import pytest

from emridispatch.config import INJECTION_KEYS, load_config
from emridispatch.parameters import NDIM, PARAM_NAMES

STUB_TABLE = {
    "off": {"I": "LISASens", "II": "LISASens"},
    "1st generation": {"A": "A1", "E": "E1", "T": "T1"},
    "2nd generation": {"A": "A2", "E": "E2", "T": "T2"},
}

TOY_INJECTION = {
    "mass_1": 1.0e+6, "mass_2": 10.0, "a": 0.0, "p": 10.0, "e": 0.1, "x": 1.0,
    "q_k": 1.0, "phi_k": 1.5707963267948966,
    "q_s": 1.0, "phi_s": 1.5707963267948966,
    "luminosity_distance": 1.0, "phi_phi": 1.5707963267948966,
    "phi_theta": 1.5707963267948966, "phi_r": 1.5707963267948966,
}


def _injection_block():
    """The injection section for whatever INJECTION_KEYS currently holds.

    A parameter added to the schema gets a placeholder here so the toy fixtures
    keep working; the shipped configs under examples/ still have to be updated
    by hand, and test_every_example_config_loads says so.
    """
    return "\n".join(f"  {name}: {TOY_INJECTION.get(name, 0.5)!r}"
                     for name in INJECTION_KEYS)


MINIMAL_TOY_CONFIG = f"""
injection:
{_injection_block()}

data:""" + """
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
LNPRIOR = -1.0


def make_run_dir(tmp_path, with_truth=True, with_summary=True, with_spec=False,
                 temps=None, ladder=None, dead_row=None):
    """Fake impulse run dir: chain_N.txt per rung + shared sidecars.

    The lnprob column follows impulse's tempered convention,
    lnprior + lnlike/temp, so converters can be checked against a known
    constant lnprior. Pass `temps` as a per-rung (NSTEPS,) array to emulate a
    ladder that adapted mid-run, `ladder` to override the rung temperatures
    (build_ladder always ends at np.inf, so pass one to get a beta=0 rung), and
    `dead_row` to set lnlike = -inf on that row of every rung, as an
    out-of-domain waveform does.
    """
    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rung_temps = TEMPS if ladder is None else ladder
    for rung, temp in enumerate(rung_temps):
        temp_col = np.full(NSTEPS, temp) if temps is None else np.asarray(temps[rung])
        cols = np.empty((NSTEPS, NDIM + 4))
        cols[:, :NDIM] = TRUTH + 0.05 * rng.standard_normal((NSTEPS, NDIM))
        cols[:, NDIM] = -0.5 * rng.random(NSTEPS)      # lnlike
        if dead_row is not None:
            cols[dead_row, NDIM] = -np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[:, NDIM + 1] = LNPRIOR + cols[:, NDIM] / temp_col   # lnprob
        cols[:, NDIM + 2] = 1.0                        # accepted
        cols[:, NDIM + 3] = temp_col
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


def make_eryn_run_dir(tmp_path, adapt_betas=False, dead_row=None):
    """Synthetic eryn_chain.h5 mirroring eryn's HDFBackend layout, with an
    over-allocated tail beyond `iteration` and a beta=0 top rung.

    adapt_betas drifts the middle of the ladder step by step, as eryn's
    adaptive tempering does, while pinning the cold rung at beta=1. dead_row
    sets log_like = -inf on that step of every rung, as an out-of-domain
    waveform does.
    """
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
    if adapt_betas:
        betas[:ERYN_IT, 1] = np.linspace(0.2, 0.5, ERYN_IT)
    if dead_row is not None:
        log_like[dead_row] = -np.inf
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


def make_eryn_untempered_run_dir(tmp_path):
    """Synthetic eryn_chain.h5 for an ntemps=1 run, the config default.

    eryn builds no TemperatureControl for a single rung, so its State carries
    no betas and save_step never writes the dataset; it stays at the HDF5 fill
    value of 0 for every stored step, exactly as reproduced here.
    """
    import h5py

    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    chain = rng.standard_normal((ERYN_IT, 1, ERYN_NW, 1, NDIM))

    with h5py.File(tmp_path / "eryn_chain.h5", "w") as f:
        g = f.create_group("mcmc")
        g.attrs["iteration"] = ERYN_IT
        g.attrs["ntemps"] = 1
        g.attrs["nwalkers"] = ERYN_NW
        g.create_dataset("chain/model_0", data=chain)
        g.create_dataset("log_like", data=np.zeros((ERYN_IT, 1, ERYN_NW)))
        g.create_dataset("log_prior", data=np.zeros((ERYN_IT, 1, ERYN_NW)))
        g.create_dataset("betas", data=np.zeros((ERYN_IT, 1)))
        g.create_dataset("accepted", data=np.full((1, ERYN_NW), 2.0))
        g.create_dataset("swaps_accepted", data=np.zeros(0))

    summary = {"config": {"backend": "eryn", "ntemps": 1,
                          "nwalkers": ERYN_NW}}
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    (tmp_path / "config.yaml").write_text(
        "run:\n  seed: 42\nsampler:\n  backend: eryn\n")
    return tmp_path
