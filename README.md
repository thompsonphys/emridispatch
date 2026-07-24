# emridispatch

Feel like running parameter estimation on extreme mass-ratio inspiral (EMRI)
gravitational-wave signals? This could provides an easy-to-use YAML-based interface
to inject and recover EMRI signals using (eventually) a broad range of Bayesian
samplers. There is also flexibility to use a variety of TDI/response
implementations, Fisher matrix providers, and flexible per-parameter priors.

By default the code samples the 12-D EMRI vector (currently equatorial-orbit support)
`[ln m1, ln m2, a, p, e, dist, q_s, phi_s, q_k, phi_k, phi_phi, phi_r]`
against a matched-filter likelihood built from one injection, with
Fisher-sized intrinsic prior boxes, an optional whitening reparametrization of
the intrinsic block, PT-MCMC with adaptive mode-jump proposals, multi-chain
R-hat drivers, and a P-P calibration harness (should you be brave enough to 
spend all of your GPU resources on this).

## Install

With pip:

```bash
pip install -e .                       # core only (numpy/scipy/pyyaml)
pip install -e ".[impulse]"            # + the impulse PT-MCMC backend
pip install -e ".[eryn]"               # + the eryn PT ensemble backend
pip install -e ".[lisatools]"          # + lisa-analysis-tools/FEW response
pip install -e ".[fisher]"             # + StableEMRIFisher
pip install -e ".[all]"                # everything
```

Or with [uv](https://docs.astral.sh/uv/) (creates/uses `.venv` automatically):

```bash
uv venv                                # create .venv (once)
uv pip install -e .                    # core only
uv pip install -e ".[all]"             # everything (same extras as above)
```

Everything heavy is optional and imported lazily; a missing extra produces a
clear install hint at the point of use. If e.g. `impulse-mcmc` / `stableemrifisher` are
not on PyPI in your environment, install them from their git sources first.

## Run

```bash
emridisp examples/my_config.yaml                   # single PE run
emridisp-multichain my_config.yaml --nchains 4     # independent chains -> R-hat
emridisp-diagnostics chains_seed1 chains_seed2     # convergence report
emridisp-pp my_config.yaml --nruns 20              # P-P calibration test
```

Everything is configured through the YAML file (see `examples/emri_config_full.yaml`
for the annotated reference). Logs go to the console and `<outdir>/run.log` by default.

## Postprocess & visualize

```bash
emridisp-postprocess my_outdir              # raw backend output -> results.h5
emridisp-plot my_outdir                     # corner + 1D marginals (posterior)
emridisp-plot my_outdir --all-temps         # overlay the full temperature ladder
emridisp-plot my_outdir --temps 0 3 5 --burn 500
```

`emridisp-postprocess` converts a run's backend-specific raw output (impulse
`chain_N.txt` files or eryn's `eryn_chain.h5`, walkers flattened per rung) into a
self-contained `results.h5`. This file contains chains 
for every temperature rung in both sampling
and physical coordinates, the injection truth, prior spec + bounds, the run config,
plus verbatim copies of the run's `config.yaml` and `run.log`, ready to pass into
`emridisp-plot` for visualization. 
The stored prior spec reconstructs the exact `JointPrior`
(`Results.load(path).prior()`) for prior draws or reweighting in
postprocessing. Conversion needs the `results` extra (h5py); plotting needs
`viz` (h5py, matplotlib, corner).

## Architecture

```
config.yaml -> build_problem(cfg) -> SamplingProblem -> backend.run(problem, cfg)
```

- **`emridispatch.pipeline`** builds a sampler-agnostic `SamplingProblem`: the raw
  physical-space likelihood, a structured `JointPrior`, the reparam transform,
  start point, and proposal covariance. No sampler is imported.
- **`emridispatch.backends`** — sampler registry. Built in: `impulse` (PT-MCMC,
  temperature ladder, cross-chain mode jumps; `chain_N.txt` output) and `eryn`
  (parallel-tempered ensemble sampler, stretch move, `eryn_chain.h5` output).
  Shared sampler knobs live at the top of the config `sampler:` section;
  backend-specific ones in its `impulse:`/`eryn:` subsections. Other sampler backends
  plug in via `register_backend()` consuming the same `SamplingProblem` —
  likelihood, TDI response, prior, and reparametrization are shared, never
  re-implemented per sampler. MCMC backends opt into the whitened coordinates
  via `problem.wrapped()`; nested-sampler backends can use the structured
  prior + physical likelihood directly.
- **`emridispatch.response`** — injection/likelihood registry (`data.response`).
  `lisatools` is the production TDI implementation; `toy` is a dependency-free
  Gaussian for structure testing. Other TDI codes register via
  `register_model()`.
- **`emridispatch.fisher`** — Fisher providers (`prior.fisher`):
  `sef` (StableEMRIFisher), `manual` (config sigmas / covariance npz),
  `deltas` (rectangular box prior `truth ± box_scale·delta` from configured
  half-widths), or a loud heuristic fallback so the pipeline runs 
  end-to-end with nothing installed.
- **`emridispatch.priors`** — per-parameter distributions (uniform, log-uniform,
  Gaussian, sine/cosine, periodic, user callables) composed into a
  `JointPrior`; override any parameter from the config `priors:` section.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The end-to-end test uses the toy response + heuristic Fisher (the impulse run
test is skipped when `impulse-mcmc` is not installed).
