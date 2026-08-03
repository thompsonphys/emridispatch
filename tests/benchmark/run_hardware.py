"""End-to-end tests against the real FEW / lisatools / StableEMRIFisher stack.

Not part of pytest; each case runs in its own subprocess so
CUDA_VISIBLE_DEVICES selects device for both FEW and lisatools.
Usage: python tests/benchmark/run_hardware.py [case ...] [--list] [--keep]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

DURATION = 0.1
DELTA_T = 10.0

SAMPLER_CONFIG = """
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
  phi_theta: 0.0
  phi_r: 1.5707963267948966

data:
  response: lisatools
  duration: {duration}
  delta_t: {delta_t}
  inj_snr: 30.0
  channels: [A, E]
  tdi: 2nd

sampler:
{sampler_block}

prior:
  fisher: manual
  box_scale: 25.0
  fisher_use_gpu: false
  sigmas:
    mass_1: 1.0
    mass_2: 1.0e-4
    a: 1.0e-5
    p: 1.0e-6
    e: 1.0e-6
    luminosity_distance: 0.05

reparam:
  mode: auto
  idx: [0, 1, 2, 3, 4, 5]

run:
  seed: 1245
  outdir: {outdir}
"""

IMPULSE_BLOCK = """  backend: impulse
  nsamples: 60
  impulse:
    cov_update: 30
    save_freq: 30
    ladder:
      max_temp: 100.0
      t_split: 10.0
      ntemps_low: 2
      ntemps_high: 1
    mode_jump:
      method: none
      weight: 25.0"""

ERYN_BLOCK = """  backend: eryn
  nsamples: 4
  eryn:
    nwalkers: 24
    ntemps: 1
    burn: 0
    thin_by: 1
    progress: false
    start_spread: 1.0"""

CASES = {
    "impulse-cpu": dict(kind="sampler", device="cpu", block=IMPULSE_BLOCK),
    "impulse-gpu": dict(kind="sampler", device="gpu", block=IMPULSE_BLOCK),
    "eryn-cpu": dict(kind="sampler", device="cpu", block=ERYN_BLOCK),
    "eryn-gpu": dict(kind="sampler", device="gpu", block=ERYN_BLOCK),
    "sef-fisher-cpu": dict(kind="fisher", device="cpu"),
}


# ---------------------------------------------------------------------------
# case bodies (run inside the child process)
# ---------------------------------------------------------------------------

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _expected_backend_tag(device):
    return "cpu" if device == "cpu" else "cuda"


def _check_backend_log(logpath, device):
    with open(logpath) as fh:
        lines = [ln for ln in fh if "backends: response=" in ln]
    _assert(lines, f"no 'backends:' line in {logpath}")
    line = lines[-1].strip()
    tag = _expected_backend_tag(device)
    _assert(tag in line.split("backends:")[-1],
            f"expected {tag!r} backends, got: {line}")
    return line


def run_sampler_case(name, spec, outdir):
    from emridispatch.cli import set_env_guards

    set_env_guards()

    import numpy as np

    from emridispatch.config import load_config
    from emridispatch.logging_utils import setup_logging
    from emridispatch.pipeline import run_from_config

    cfg_path = os.path.join(outdir, "config.yaml")
    os.makedirs(outdir, exist_ok=True)
    with open(cfg_path, "w") as fh:
        fh.write(SAMPLER_CONFIG.format(
            duration=DURATION, delta_t=DELTA_T, outdir=outdir,
            sampler_block=spec["block"]))

    setup_logging(outdir=outdir, level="INFO", filename="run.log")
    cfg = load_config(cfg_path)
    summary = run_from_config(cfg, resume=False)

    backend_line = _check_backend_log(os.path.join(outdir, "run.log"),
                                      spec["device"])

    truth_path = os.path.join(outdir, "injection_truth.json")
    _assert(os.path.exists(truth_path), "injection_truth.json not written")
    truth = json.loads(open(truth_path).read())
    _assert(len(truth["sampling_vector"]) == 12,
            f"sampling vector is {len(truth['sampling_vector'])}-D, expected 12")

    if cfg.sampler.backend == "impulse":
        chain = np.loadtxt(os.path.join(outdir, "chain_0.txt"))
        _assert(chain.ndim == 2 and chain.shape[0] > 0, "empty chain_0.txt")
        _assert(chain.shape[1] == 16, f"chain has {chain.shape[1]} columns")
        lnlike = chain[:, 12]
        nsteps = chain.shape[0]
    else:
        import h5py

        with h5py.File(os.path.join(outdir, "eryn_chain.h5"), "r") as f:
            g = f["mcmc"]
            it = int(g.attrs["iteration"])
            lnlike = np.asarray(g["log_like"][:it]).ravel()
            samples = np.asarray(g["chain/model_0"][:it])
        _assert(it > 0, "eryn wrote zero iterations")
        _assert(samples.shape[-1] == 12,
                f"eryn chain is {samples.shape[-1]}-D, expected 12")
        _assert(np.isfinite(samples).all(), "non-finite samples in eryn chain")
        nsteps = it

    _assert(np.isfinite(lnlike).all(), "non-finite log-likelihood in chain")
    _assert(lnlike.max() > -np.inf, "no finite log-likelihood")

    return {
        "backends": backend_line.split("backends:")[-1].strip(),
        "steps": nsteps,
        "max_lnlike": float(np.max(lnlike)),
        "proposal_acceptance": _acceptance(summary),
    }


def _acceptance(summary):
    import numpy as np

    if summary is None:
        return None
    acc = summary.get("proposal_acceptance")
    if acc is None:
        return None
    if isinstance(acc, dict):
        return {k: (round(float(v["rate"]), 4) if isinstance(v, dict) else
                    round(float(np.mean(np.asarray(v, dtype=float))), 4))
                for k, v in acc.items()}
    return round(float(np.mean(np.asarray(acc, dtype=float))), 4)


def run_fisher_case(name, spec, outdir):
    from emridispatch.cli import set_env_guards

    set_env_guards()

    import logging

    import numpy as np

    from emridispatch.fisher.sef import SEFFisherProvider
    from emridispatch.parameters import INTRINSIC_ORDER

    from emridispatch.config import load_config
    from emridispatch.response import build_injection_model

    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        handlers=[logging.FileHandler(os.path.join(outdir, "run.log")),
                  logging.StreamHandler(sys.stderr)])

    cfg_path = os.path.join(outdir, "config.yaml")
    with open(cfg_path, "w") as fh:
        fh.write(SAMPLER_CONFIG.format(
            duration=DURATION, delta_t=DELTA_T, outdir=outdir,
            sampler_block=ERYN_BLOCK))
    cfg = load_config(cfg_path)

    model = build_injection_model(cfg)
    injection = dict(model.injection_parameters)
    _check_backend_log(os.path.join(outdir, "run.log"), spec["device"])

    provider = SEFFisherProvider(tdi="2nd generation", foreground=True,
                                 channels=["A", "E"])
    result = provider.compute(injection, duration=cfg.data.duration,
                              delta_t=cfg.data.delta_t, use_gpu=False)

    _assert(list(result.order) == INTRINSIC_ORDER,
            f"unexpected parameter order {result.order}")
    sig = np.array([result.sigmas[k] for k in INTRINSIC_ORDER])
    _assert(np.isfinite(sig).all(), f"non-finite sigmas: {result.sigmas}")
    _assert((sig > 0).all(), f"non-positive sigmas: {result.sigmas}")

    cov = np.asarray(result.cov)
    _assert(cov.shape == (6, 6), f"covariance shape {cov.shape}")
    _assert(np.isfinite(cov).all(), "non-finite covariance entries")
    asym = (np.abs(cov - cov.T).max()
            / max(np.abs(cov).max(), np.finfo(float).tiny))
    cond = float(np.linalg.cond(cov))
    _assert(asym < 1e3 * np.finfo(float).eps * cond,
            f"covariance asymmetry {asym:.3e} exceeds round-off "
            f"(cond={cond:.3e})")
    _assert((np.linalg.eigvalsh(0.5 * (cov + cov.T)) > 0).all(),
            "covariance is not positive definite")
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), sig, rtol=1e-10)

    corr = cov / np.outer(sig, sig)
    cond_corr = float(np.linalg.cond(corr))
    np.savez(os.path.join(outdir, "fisher.npz"), cov=cov, corr=corr, sigmas=sig,
             order=np.array(INTRINSIC_ORDER))

    evals, evecs = np.linalg.eigh(corr)
    stiff = evecs[:, 0]
    degenerate_direction = {
        INTRINSIC_ORDER[i]: round(float(stiff[i]), 4)
        for i in np.argsort(-np.abs(stiff))[:3]}
    worst = np.unravel_index(np.abs(np.triu(corr, 1)).argmax(), corr.shape)

    warnings = []
    if cond_corr > 1e13:
        warnings.append(
            f"correlation matrix cond={cond_corr:.2e} is approaching float64 "
            "resolution; the whitening transform may lose precision")
    elif cond_corr > 1e8:
        warnings.append(
            f"strongly anisotropic Fisher (corr cond={cond_corr:.2e}, stiff "
            f"direction {degenerate_direction}); intrinsic to EMRI parameter "
            "space, and the reason the whitening reparam exists")
    frac = {k: float(result.sigmas[k] / abs(injection[k]))
            for k in INTRINSIC_ORDER if injection.get(k)}
    loose = {k: round(v, 3) for k, v in frac.items() if v > 1.0}
    if loose:
        warnings.append(f"fractional sigma > 1 for {loose}")

    return {
        "snr": float(model.optimal_snr),
        "sigmas": {k: float(v) for k, v in result.sigmas.items()},
        "frac_sigma": {k: round(v, 4) for k, v in frac.items()},
        "cond": cond,
        "cond_corr": cond_corr,
        "max_corr": float(np.abs(corr - np.eye(6)).max()),
        "max_corr_pair": [INTRINSIC_ORDER[worst[0]], INTRINSIC_ORDER[worst[1]],
                          round(float(corr[worst]), 6)],
        "corr_eigenvalues": [float(v) for v in evals],
        "degenerate_direction": degenerate_direction,
        "asymmetry": float(asym),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def child_main(name, outdir):
    spec = CASES[name]
    runner = run_sampler_case if spec["kind"] == "sampler" else run_fisher_case
    info = runner(name, spec, outdir)
    print("BENCHMARK_RESULT " + json.dumps(info))


def _extract_error(proc):
    """Pull the traceback out of the child's output.

    Filters nanobind's post-GPU-run `leaked` noise that buries it.
    """
    lines = [ln for ln in (proc.stderr or "").splitlines()
             if not ln.startswith("nanobind:") and not ln.startswith(" - leaked")
             and "skipped remainder" not in ln
             and "refleaks.html" not in ln]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback")]
    if starts:
        return "\n".join(lines[starts[-1]:])
    return "\n".join(lines[-25:]) or f"exit {proc.returncode}"


def run_case(name, outdir, keep):
    spec = CASES[name]
    env = dict(os.environ)
    if spec["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    cmd = [sys.executable, os.path.abspath(__file__), "--child", name,
           "--outdir", outdir, "--duration", str(DURATION),
           "--delta-t", str(DELTA_T)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
    elapsed = time.time() - t0

    info, err = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("BENCHMARK_RESULT "):
            info = json.loads(line[len("BENCHMARK_RESULT "):])
    if proc.returncode != 0 or info is None:
        err = _extract_error(proc)
    logpath = os.path.join(outdir, "benchmark.out")
    with open(logpath, "w") as fh:
        fh.write(proc.stdout + "\n===== stderr =====\n" + proc.stderr)
    return dict(name=name, ok=err is None, elapsed=elapsed, info=info,
                error=err, log=logpath if (keep or err) else None)


def main():
    global DURATION, DELTA_T

    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--outroot", default=None)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--delta-t", type=float, default=DELTA_T)
    ap.add_argument("--child")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    DURATION = args.duration
    DELTA_T = args.delta_t

    if args.child:
        child_main(args.child, args.outdir)
        return 0

    if args.list:
        for name, spec in CASES.items():
            print(f"{name:16s} {spec['kind']:8s} {spec['device']}")
        return 0

    names = args.cases or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}; known: {list(CASES)}")

    outroot = args.outroot or tempfile.mkdtemp(prefix="emridispatch-benchmark-")
    os.makedirs(outroot, exist_ok=True)
    print(f"benchmark run dirs: {outroot}\n")

    results = []
    for name in names:
        print(f"[running] {name} ...", flush=True)
        res = run_case(name, os.path.join(outroot, name), args.keep)
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        print(f"[{status:4s}] {name}  {res['elapsed']:.1f}s")
        if res["ok"]:
            print(textwrap.indent(json.dumps(res["info"], indent=2), "    "))
        else:
            print(textwrap.indent(res["error"], "    "))
        print(flush=True)

    print("=" * 60)
    for res in results:
        print(f"{'PASS' if res['ok'] else 'FAIL'}  {res['name']:16s} "
              f"{res['elapsed']:7.1f}s")
    nfail = sum(not r["ok"] for r in results)
    print(f"{len(results) - nfail}/{len(results)} passed  (logs under {outroot})")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
