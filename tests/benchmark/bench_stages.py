"""Per-stage timing profile of the EMRI waveform -> response -> likelihood path.

Run on two machines and diff the JSON to find where a GPU is losing time.
Separates the three things that get conflated in an end-to-end wall time:
FEW waveform generation, the TDI response, and the likelihood inner products --
plus a forced-CPU waveform, which measures the host CPU rather than the device.

Usage:
    python tests/benchmark/bench_stages.py                      # default sweep
    python tests/benchmark/bench_stages.py --durations 0.1 0.5 2.0 --repeat 5
    python tests/benchmark/bench_stages.py --device cpu         # CUDA_VISIBLE_DEVICES=""
    python tests/benchmark/bench_stages.py --json prof_a100.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

INJECTION = {
    "mass_1": 1.0e6, "mass_2": 10.0, "a": 0.0, "p": 10.0, "e": 0.1, "x": 1.0,
    "q_k": 1.0, "phi_k": 1.5707963267948966,
    "q_s": 1.0, "phi_s": 1.5707963267948966,
    "luminosity_distance": 1.0,
    "phi_phi": 1.5707963267948966, "phi_theta": 0.0, "phi_r": 1.5707963267948966,
}

FEW_ARGS = ("FastKerrEccentricEquatorialFlux",)
FEW_KWARGS = dict(sum_kwargs=dict(pad_output=True), return_list=False,
                  frame="detector")


def _sync():
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _free():
    """Drop cupy's cached blocks between stages.

    Each stage holds its own multi-GB waveform buffers; without this the later
    stages OOM on a consumer card even though no stage is individually large.
    """
    import gc

    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _gpu_mem():
    try:
        import cupy

        free, total = cupy.cuda.runtime.memGetInfo()
        return {"used_gb": round((total - free) / 2**30, 3),
                "total_gb": round(total / 2**30, 3)}
    except Exception:
        return None


def _time(fn, repeat):
    """Return (min, mean) seconds over `repeat` calls, after one warm-up."""
    fn()
    _sync()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return {"min": min(ts), "mean": sum(ts) / len(ts), "n": repeat}


def _params_list(params):
    return [float(params[k]) for k in
            ("mass_1", "mass_2", "a", "p", "e", "x", "luminosity_distance",
             "q_s", "phi_s", "q_k", "phi_k", "phi_phi", "phi_theta", "phi_r")]


def environment():
    env = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import few

        env["few"] = few.__version__
        env["few_backends"] = {n: few.has_backend(n)
                               for n in ("cuda12x", "cuda11x", "cpu")}
    except Exception as exc:
        env["few"] = f"unavailable: {exc}"
    try:
        import cupy

        env["cupy"] = cupy.__version__
        dev = cupy.cuda.Device()
        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(
            dev.id)["name"].decode()
        env["compute_capability"] = dev.compute_capability
        env["cupy_cache_dir"] = os.environ.get(
            "CUPY_CACHE_DIR", os.path.expanduser("~/.cupy/kernel_cache"))
        env["cupy_cache_writable"] = os.access(
            os.path.dirname(env["cupy_cache_dir"]) or ".", os.W_OK)
    except Exception as exc:
        env["cupy"] = f"unavailable: {exc}"
    query = ("name,driver_version,persistence_mode,mig.mode.current,ecc.mode.current,"
             "clocks.max.sm,clocks.sm,power.limit,memory.total,utilization.gpu")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        env["nvidia_smi"] = dict(zip(query.split(","),
                                     [v.strip() for v in
                                      out.stdout.strip().split(",")]))
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        env["gpu_processes"] = [p for p in procs.stdout.strip().splitlines() if p]
    except Exception as exc:
        env["nvidia_smi"] = f"unavailable: {exc}"
    for var in ("CUDA_VISIBLE_DEVICES", "FEW_ENABLED_BACKENDS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUPY_CACHE_DIR"):
        if var in os.environ:
            env.setdefault("env_vars", {})[var] = os.environ[var]
    return env


def bench_duration(duration, delta_t, repeat, with_cpu_waveform):
    from few.waveform.waveform import GenerateEMRIWaveform
    from lisatools.sources.emri import EMRITDIWaveform

    from emridispatch.response.lisatools import LisatoolsEMRILikelihood

    result = {"duration": duration, "delta_t": delta_t}
    params = _params_list(INJECTION)

    _free()
    t0 = time.perf_counter()
    few_gen = GenerateEMRIWaveform(*FEW_ARGS, **FEW_KWARGS)
    result["build_few"] = time.perf_counter() - t0
    result["few_backend"] = few_gen.waveform_generator.backend.name

    result["waveform_only"] = _time(
        lambda: few_gen(*params, T=duration, dt=delta_t), repeat)
    result["n_samples"] = int(
        len(few_gen(*params, T=duration, dt=delta_t)))
    del few_gen
    _free()

    if with_cpu_waveform:
        cpu_gen = GenerateEMRIWaveform(*FEW_ARGS, force_backend="cpu",
                                       **FEW_KWARGS)
        result["waveform_only_forced_cpu"] = _time(
            lambda: cpu_gen(*params, T=duration, dt=delta_t), repeat)
        del cpu_gen
        _free()

    t0 = time.perf_counter()
    tdi_gen = EMRITDIWaveform(
        T=duration, dt=delta_t,
        response_kwargs=dict(t0=30000.0, tdi="2nd generation", tdi_chan="AE"))
    result["build_response"] = time.perf_counter() - t0
    result["response_backend"] = tdi_gen.response.backend.name

    result["waveform_plus_response"] = _time(
        lambda: tdi_gen(*params), repeat)
    del tdi_gen
    _free()

    from types import SimpleNamespace

    cfg = SimpleNamespace(
        injection=dict(INJECTION),
        data=SimpleNamespace(duration=duration, delta_t=delta_t, inj_snr=30.0,
                             channels=["A", "E"], tdi="2nd generation",
                             foreground=True, add_noise=False, noise_seed=0))
    t0 = time.perf_counter()
    like = LisatoolsEMRILikelihood.from_config(cfg)
    result["build_injection"] = time.perf_counter() - t0
    result["snr"] = float(like.optimal_snr)

    import numpy as np

    inj = like.injection_parameters
    vec = np.array([
        np.log(inj["mass_1"]), np.log(inj["mass_2"]), inj["a"], inj["p"],
        inj["e"], inj["luminosity_distance"], inj["q_s"], inj["phi_s"],
        inj["q_k"], inj["phi_k"], inj["phi_phi"], inj["phi_r"]])
    result["likelihood"] = _time(lambda: like(vec), repeat)
    result["lnlike"] = float(like(vec))
    result["gpu_mem_peak"] = _gpu_mem()

    del like
    _free()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", nargs="+", type=float,
                    default=[0.1, 0.5, 2.0])
    ap.add_argument("--delta-t", type=float, default=10.0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--no-cpu-waveform", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from emridispatch.cli import set_env_guards

    set_env_guards()

    profile = {"environment": environment(), "runs": []}
    print(json.dumps(profile["environment"], indent=2), flush=True)

    for duration in args.durations:
        print(f"\n[bench] duration={duration} yr ...", flush=True)
        run = bench_duration(duration, args.delta_t, args.repeat,
                             not args.no_cpu_waveform)
        profile["runs"].append(run)
        print(f"  backends: few={run['few_backend']} "
              f"response={run['response_backend']}  n={run['n_samples']}")
        for stage in ("waveform_only", "waveform_plus_response", "likelihood",
                      "waveform_only_forced_cpu"):
            if stage in run:
                print(f"  {stage:26s} min={run[stage]['min']*1e3:9.2f} ms  "
                      f"mean={run[stage]['mean']*1e3:9.2f} ms")
        print(f"  {'build_injection':26s}     {run['build_injection']:9.2f} s")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(profile, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
