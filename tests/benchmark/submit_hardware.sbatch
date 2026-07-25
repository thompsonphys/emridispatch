#!/bin/bash
#
# Slurm driver for the emridispatch benchmark tests + GPU stage profile.
#
#   mkdir -p slurm-logs        # slurm needs the --output dir to exist already
#   sbatch tests/benchmark/submit_hardware.sbatch
#   sbatch tests/benchmark/submit_hardware.sbatch --bench-only
#   sbatch tests/benchmark/submit_hardware.sbatch --hardware-only
#
# Everything marked FILL ME is site-specific.

# ===================== FILL ME: SBATCH directives ==========================
#SBATCH --job-name=emri-bench
#SBATCH --account=FILL_ME_ACCOUNT
#SBATCH --partition=FILL_ME_GPU_PARTITION
#SBATCH --qos=FILL_ME_QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=FILL_ME_EMAIL
# #SBATCH --nodelist=FILL_ME_NODE          # pin a node to compare hardware
# #SBATCH --exclusive                      # rule out noisy co-tenants
# ===========================================================================

set -euo pipefail

# ===================== FILL ME: environment ================================
REPO="${REPO:-FILL_ME_PATH_TO/emridispatch}"
CONDA_SH="${CONDA_SH:-FILL_ME_PATH_TO/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-FILL_ME_ENV_NAME}"
# Node-local scratch. A cupy kernel cache on a network filesystem (or a
# read-only $HOME) makes every process re-JIT its kernels.
SCRATCH="${SCRATCH:-/tmp/${USER:-$(id -un)}/${SLURM_JOB_ID:-local}}"
# ===========================================================================

# Uncomment for a module-based site instead of conda:
# module purge
# module load FILL_ME_CUDA_MODULE
# module load FILL_ME_PYTHON_MODULE
# source FILL_ME_PATH_TO/venv/bin/activate

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

mkdir -p "$SCRATCH" "$REPO/slurm-logs"
export CUPY_CACHE_DIR="$SCRATCH/cupy_cache"
mkdir -p "$CUPY_CACHE_DIR"

# FEW's generator is not thread-safe and the likelihood is called serially, so
# keep the host libraries single-threaded (emridispatch.cli.set_env_guards does
# this in-process too; setting it here covers the child processes as well).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"

MODE="${1:-all}"
TAG="${SLURM_JOB_ID:-local}-$(hostname -s)"
OUTROOT="$SCRATCH/benchmark-$TAG"
RESULTS="$REPO/slurm-logs/results-$TAG"
mkdir -p "$OUTROOT" "$RESULTS"

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-none}  ${SLURM_JOB_NAME:-}"
echo "node       : $(hostname -f)"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?}"
echo "gpus       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "repo       : $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git repo'))"
echo "scratch    : $SCRATCH"
echo "results    : $RESULTS"
echo "=============================================================="

# ---- device inventory: MIG slice, clock caps, ECC, co-tenants -------------
echo "--- nvidia-smi ---"
nvidia-smi || echo "nvidia-smi unavailable"
nvidia-smi --query-gpu=name,driver_version,persistence_mode,mig.mode.current,\
ecc.mode.current,clocks.max.sm,clocks.sm,power.limit,power.draw,memory.total \
    --format=csv | tee "$RESULTS/nvidia-smi.csv" || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    | tee "$RESULTS/gpu-processes.csv" || true
echo "--- lscpu ---"
lscpu | tee "$RESULTS/lscpu.txt" | grep -E "Model name|MHz|Socket|Core|Thread" || true

# ---- preflight: cupy loads its CUDA libs lazily, so a missing libcufft only
# ---- surfaces at the first FFT, minutes into a run. Probe each one up front.
if [[ "${SKIP_PREFLIGHT:-0}" != 1 && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "--- cupy preflight ---"
    if ! python - <<'PY' 2>&1 | tee "$RESULTS/preflight.log"
import sys

import cupy as cp

print("cupy", cp.__version__, "/ runtime", cp.cuda.runtime.runtimeGetVersion())
print("device", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())

checks = {
    "cudart": lambda: cp.arange(8.0) + 1,
    "cublas": lambda: cp.eye(4) @ cp.eye(4),
    "cufft": lambda: cp.fft.rfft(cp.arange(8.0)),
    "cusolver": lambda: cp.linalg.cholesky(cp.eye(4)),
    "curand": lambda: cp.random.standard_normal(8),
    "nvrtc": lambda: cp.ElementwiseKernel("float64 x", "float64 y", "y = x * 2", "pf")(
        cp.arange(8.0)
    ),
}

bad = []
for name, fn in checks.items():
    try:
        fn()
        cp.cuda.Stream.null.synchronize()
        print(f"  {name:9s} ok")
    except Exception as exc:
        bad.append(name)
        print(f"  {name:9s} FAIL  {type(exc).__name__}: {exc}".splitlines()[0])

if bad:
    print(f"unusable GPU stack: {', '.join(bad)}")
    sys.exit(1)
PY
    then
        echo
        echo "preflight FAILED -- aborting before the benchmark burns walltime."
        echo "install the missing CUDA 12 runtime libs into \$CONDA_ENV, e.g."
        echo "  conda install -n $CONDA_ENV -c conda-forge libcufft libcusolver libcurand"
        echo "  # or: pip install nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12"
        echo "re-run with SKIP_PREFLIGHT=1 to bypass this check."
        exit 1
    fi
fi

# ---- sample SM utilization for the whole job ------------------------------
# Sustained SM% below ~20 during the waveform stages means the GPU is
# launch-latency bound, not compute bound.
DMON_PID=""
if command -v nvidia-smi >/dev/null; then
    nvidia-smi dmon -s ucm -d 1 -o T > "$RESULTS/dmon.txt" 2>/dev/null &
    DMON_PID=$!
fi
cleanup() { [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true; }
trap cleanup EXIT

STATUS=0

# ---- stage profile: waveform / +response / likelihood, GPU and forced CPU --
if [[ "$MODE" == "all" || "$MODE" == "--bench-only" ]]; then
    echo
    echo "############ stage profile (GPU) ############"
    python tests/benchmark/bench_stages.py \
        --durations 0.1 0.5 2.0 \
        --repeat 5 \
        --json "$RESULTS/profile-gpu.json" || STATUS=1

    echo
    echo "############ stage profile (CPU baseline) ############"
    # Same code path with CUDA_VISIBLE_DEVICES="" -- isolates host CPU speed,
    # which is the usual reason a cluster node loses to a desktop.
    python tests/benchmark/bench_stages.py \
        --device cpu \
        --durations 0.1 \
        --repeat 3 \
        --json "$RESULTS/profile-cpu.json" || STATUS=1
fi

# ---- end-to-end hardware cases ----------------------------------------------
if [[ "$MODE" == "all" || "$MODE" == "--hardware-only" ]]; then
    echo
    echo "############ hardware tests ############"
    python tests/benchmark/run_hardware.py \
        --keep --outroot "$OUTROOT" \
        2>&1 | tee "$RESULTS/hardware.log" || STATUS=1
fi

cleanup
trap - EXIT

echo
echo "=============================================================="
echo "artifacts in $RESULTS:"
ls -la "$RESULTS"
echo "run dirs kept in $OUTROOT"
echo "exit status: $STATUS"
echo "=============================================================="

# Compare two machines:
#   python -m json.tool results-<a100>/profile-gpu.json
#   python -m json.tool results-<other>/profile-gpu.json
# Diff the per-stage min times and the "environment" block.

exit $STATUS
