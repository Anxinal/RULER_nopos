#!/bin/bash
# =============================================================================
# run_experiments.sh
#
# End-to-end experiment runner for MaskedTransformer on the RULER benchmark.
#
# Pipeline (per configuration):
#   1. Generate RULER training data   (data/prepare.py, seed=0)
#   2. Train MaskedTransformer        (tmodel/train.py)
#   3. Generate RULER eval data       (data/prepare.py, seed=42)
#   4. Run predictions                (pred/call_api.py --server_type tmodel)
#   5. Compute metrics                (eval/evaluate.py)
#
# Modes
# -----
#   bash run_experiments.sh              # submit to SLURM
#   bash run_experiments.sh --dry-run    # print commands only
#   bash run_experiments.sh --local      # run sequentially, no SLURM
#   bash run_experiments.sh --summary    # collect results into CSV
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====================== CLUSTER (edit for your site) =========================
PARTITION="${PARTITION:-gpu}"
# Full Slurm GPU spec, "<type>:<count>". The type selector is not cosmetic: with a
# bare count, jobs land on whatever GPU the partition offers, and a torch wheel
# without kernels for that architecture fails only once training starts. Pinning the
# type also keeps every cell of the sweep on identical hardware, which comparing
# arms against each other depends on.
GPU_SPEC="${GPU_SPEC:-h100-96:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-12:00:00}"
# Checkpoints are deleted once a cell has been evaluated at every length, since
# the predictions and summaries are what the analysis reads. At ~0.9 GB per cell
# this is the difference between ~8 GB and ~0 GB of standing disk. Set true to keep
# them, e.g. to re-run evaluation later without retraining.
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-false}"
# Reuse an existing checkpoint instead of retraining, so a cell whose evaluation was cut
# short (a TIME kill lands before the checkpoint is deleted) can be resubmitted and go
# straight to prediction. call_api.py resumes by sample index, so prediction continues
# where it stopped rather than starting over. RETRAIN=true forces training regardless.
RETRAIN="${RETRAIN:-false}"
# Python environment. The virtualenv is created on first use and populated from
# REQUIREMENTS. It must live somewhere every compute node can see, which the repo
# root normally is; override VENV_DIR if your home is not shared.
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/../.venv}"
REQUIREMENTS="${REQUIREMENTS:-${SCRIPT_DIR}/../requirements.txt}"
# Interpreter used to build the venv, not the one inside it.
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
# Install missing Python packages from inside the job rather than by hand first.
# torch is always excluded; see check_environment for why.
AUTO_INSTALL="${AUTO_INSTALL:-true}"
# Extra flags for every pip install, for sites where PyPI is not directly reachable
# from a compute node. Examples:
#   PIP_ARGS='--index-url https://<internal-mirror>/simple'
#   PIP_ARGS='--no-index --find-links $HOME/wheels'
PIP_ARGS="${PIP_ARGS:-}"
# torch is never installed from a bare `pip install torch`, because a plain PyPI wheel
# may be CPU-only or built against the wrong CUDA and the failure is silent. Name the
# exact wheel for this cluster and the job will install it, e.g.
#   TORCH_SPEC='torch --index-url https://download.pytorch.org/whl/cu121'
TORCH_SPEC="${TORCH_SPEC:-}"

# ====================== MODEL ================================================
# 286M -> ~70M. Capacity was never the binding constraint: the arm that works reached
# 98 at the larger size, and a <1M-parameter reproduction of the same task structure
# reached 83-100%. Excess capacity actively works against us here -- 286M parameters
# against 67.5k unique samples is heavy pressure to memorise the answer FORMAT, which
# is precisely the basin every failing cell settled into (ppl ~49, 3-6 distinct outputs
# for 1000 questions). Less room to memorise biases toward the general solution, and
# ~4x cheaper cells make the remaining open questions answerable in hours not days.
#
# NUM_HEADS stays 8: the mask specs are one code per head, and CCCCFFFF's 4-causal /
# 4-future split is the object under study. head_dim is 512/8 = 64, the usual value.
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=6
D_FF=2048
# 0.0, not 0.2. embed_dropout is applied to the SUM of the token embedding and the
# additive positional encoding, so for the sinusoidal arms it randomly deletes 20% of a
# vector in which content and position are already entangled -- a penalty RoPE and ALiBi
# never pay, since their positional signal lives in the attention scores and is never
# dropped. Both configurations known to learn this task (the --sanity control, and the
# small-scale reproduction) ran at 0.0. Raise it only if val loss starts diverging from
# train; with random per-sample needles there is little here to overfit.
DROPOUT=0.0
MAX_LEN=16384     # PE buffer length (must be >= longest EVAL seq length)
TOKENIZER="gpt2"

# ====================== DATA =================================================
# Length-generalisation design: train at one length, evaluate at multiples of it.
# Every eval length must stay <= MAX_LEN, which sizes the sinusoidal and rotary
# buffers; exceeding it now raises a clear error rather than failing obscurely past
# the end of the table.
#
# Train and eval data are generated by separate prepare.py runs into separate trees,
# so these two lengths are independent. Training short and evaluating long is the
# point of the experiment, and it is also much cheaper: cost per sample grows with
# the square of the sequence length in attention.
TRAIN_SEQ_LENGTHS=(2048)    # seq lengths for training data
# Samples per task per seq length -- PER TASK, so five tasks train on 125k samples. For
# the QA tasks this exceeds the unique training questions (see QA_HOLDOUT), so each is
# reused a few times with a different draw of distractor paragraphs.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-25000}"   # samples per task per seq length
# MUST differ from EVAL_SEED. Both were 42, which made the two prepare.py runs produce
# identical RNG streams at the same --max_seq_length, so every eval sample at the train
# length was literally a training sample and that whole column measured memorisation.
TRAIN_SEED=1234

# 2048 is the training length, then 2x and 4x it.
EVAL_SEQ_LENGTHS=(2048 4096 8192)
EVAL_SAMPLES=1000
# QA questions reserved for evaluation. SQuAD and HotpotQA are fixed pools (~5.9k and
# ~7.4k answerable questions), and qa.py takes questions IN ORDER from the start of the
# pool -- the seed only reshuffles distractor paragraphs, never which questions appear.
# Without a split, train and eval both start at question 0 and training cycles the whole
# pool, so every eval question and its gold answer is a training target, and a model can
# score by recalling question -> answer without reading the context. Eval takes
# [0, QA_HOLDOUT), training takes [QA_HOLDOUT, end). The margin over EVAL_SAMPLES absorbs
# questions skipped for not fitting the length budget.
QA_HOLDOUT=2000
EVAL_SEED=42                # RULER default

# ====================== TRAINING =============================================
EPOCHS=30
BATCH_SIZE=8
GRAD_ACCUM=8                # effective batch = BATCH_SIZE * GRAD_ACCUM
LR=3e-4                     # the --sanity control that passes uses 3e-4
WARMUP=1000
# Early stopping. EPOCHS is a cap, not a target: a cell stops once val loss has failed
# to beat its running best by MIN_DELTA for PATIENCE consecutive epochs.
#
# Note MIN_DELTA is measured against the running BEST, not the previous epoch, so the
# bar ratchets upward. An epoch can write a new best.pt and still increment the stale
# counter, if it improved by less than MIN_DELTA.
#
# These gate on pooled val loss, which is exactly the quantity the degenerate solution
# already optimises: a model that memorises the answer FORMAT and ignores the context
# settles at ppl ~49 and sits there, val loss flattens, and patience expires while the
# retrieval circuit has not begun to form. The previous settings certified that basin as
# "converged" at epoch 17 in every failing cell. The real remedy is MIN_LR_FRAC below --
# those cells were at 5e-5 by the time they stopped, far too small to escape.
#
# MIN_EPOCHS only suppresses the break; the stale counter keeps climbing underneath it.
# So an arm that plateaus early stops at exactly MIN_EPOCHS, not MIN_EPOCHS + PATIENCE.
# The floor only bites when MIN_EPOCHS > PATIENCE -- otherwise reaching stale >= PATIENCE
# already implies that many epochs have elapsed and patience alone governs. At 12 and 8
# the floor is live: a cell flat from the start burns patience at epoch 8 and is then
# held to epoch 12 before it may stop.
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-5e-3}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-12}"
# Floor the cosine schedule at this fraction of the peak LR rather than decaying to 0.
# Escape from the format basin is a circuit formation, not a smooth descent, so it needs
# a step size large enough to explore. The arm that solved the task escaped at epoch 8
# with LR near peak; failing arms were at 5e-5 by the time they stopped.
MIN_LR_FRAC="${MIN_LR_FRAC:-0.25}"
SRC_LEN=2048                # max encoder tokens during training
TGT_LEN=128                # max decoder tokens during training
SEED=42

# ====================== FLAGS ================================================
# Parsed before the grid is built, so --sanity can override the configuration above
# before it is validated.
DRY_RUN=false
LOCAL=false
SUMMARY=false
SANITY=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --local)   LOCAL=true ;;
        --summary) SUMMARY=true ;;
        --sanity)  SANITY=true ;;
    esac
done

# ====================== EXPERIMENT GRID ======================================
# Per-head encoder mask specs: one code per attention head (B/C/F), so each entry
# must be exactly NUM_HEADS characters long. Head order carries no meaning -- heads
# are concatenated and mixed by one output projection, so only the count of each
# code matters. The decoder is always causal and is not an axis of this grid.
MASK_CONFIGS=("BBBBBBBB" "CCCCCCCC" "CCCCFFFF")

# The all-bidirectional spec is the "no mask" condition: every head attends
# everywhere, so position can only come from the encoding.
NO_MASK_SPEC="BBBBBBBB"

# This is NOT a full cross, deliberately. A cell with both a mask and a positional
# encoding confounds the two sources of position, and the question is which one
# supplies it. So:
#

PE_CROSSED_WITH_MASKS=("none" "sinusoidal")
# "learned" is dropped. train.py still implements it; it is simply not an arm here.
PE_NO_MASK_ONLY=("rope" "alibi")

# ====================== SANITY MODE ==========================================
# A positive control. One task, short context, small model, evaluated at the length
# it trained on, with a score it must beat.
#
# This exists because a null result is uninterpretable without one. Three separate
# defects -- an initialisation that started the loss near 1000, a target tokenisation
# that made the answer uncopyable, and training on one of ten answers -- all produced
# the same signal: zeros in every cell. None was distinguishable from "masks cannot
# substitute for positional encodings", which is the thing the grid is meant to
# measure. Run this before spending GPU-days on the grid.
SANITY_MIN_SCORE="${SANITY_MIN_SCORE:-40}"

if $SANITY; then
    echo "=== SANITY MODE: positive control, not an experiment ==="
    SANITY_TASKS=("niah_single_1")     # single needle, noise haystack, no corpus needed
    TRAIN_SEQ_LENGTHS=(512)
    EVAL_SEQ_LENGTHS=(512)             # in-distribution on purpose
    TRAIN_SAMPLES="${SANITY_TRAIN_SAMPLES:-4000}"
    EVAL_SAMPLES="${SANITY_EVAL_SAMPLES:-200}"

    D_MODEL=256; NUM_HEADS=4; NUM_LAYERS=4; D_FF=1024; DROPOUT=0.0
    MAX_LEN=1024; SRC_LEN=512; TGT_LEN=32

    EPOCHS="${SANITY_EPOCHS:-30}"; BATCH_SIZE=16; GRAD_ACCUM=1
    LR=3e-4; WARMUP=200
    KEEP_CHECKPOINTS=true              # keep it, so a failure can be inspected

    # Sinusoidal + causal: the arm most likely to work. If this cannot retrieve a
    # needle from 512 tokens it saw in training, nothing in the grid is meaningful.
    SANITY_SPEC="$(printf 'C%.0s' $(seq 1 "${NUM_HEADS}"))"
    MASK_CONFIGS=("${SANITY_SPEC}")
    NO_MASK_SPEC="$(printf 'B%.0s' $(seq 1 "${NUM_HEADS}"))"
    PE_CROSSED_WITH_MASKS=("sinusoidal")
    PE_NO_MASK_ONLY=()
fi


for spec in "${MASK_CONFIGS[@]}"; do
    if [ ${#spec} -ne "${NUM_HEADS}" ]; then
        echo "ERROR: mask spec '${spec}' has ${#spec} codes but NUM_HEADS=${NUM_HEADS}." >&2
        exit 1
    fi
done
if [ "${NO_MASK_SPEC}" != "$(printf 'B%.0s' $(seq 1 "${NUM_HEADS}"))" ]; then
    echo "ERROR: NO_MASK_SPEC='${NO_MASK_SPEC}' is not all-bidirectional for NUM_HEADS=${NUM_HEADS}." >&2
    exit 1
fi

# Flatten the grid into explicit "<pe> <mask_spec>" cells.
EXPERIMENTS=()
for pe in "${PE_CROSSED_WITH_MASKS[@]}"; do
    for spec in "${MASK_CONFIGS[@]}"; do
        EXPERIMENTS+=("${pe} ${spec}")
    done
done
for pe in ${PE_NO_MASK_ONLY[@]+"${PE_NO_MASK_ONLY[@]}"}; do
    EXPERIMENTS+=("${pe} ${NO_MASK_SPEC}")
done

# Task list (must match entries in synthetic.yaml)
source "${SCRIPT_DIR}/config_tasks.sh"
# An explicit include list, not the full synthetic[@] suite: the two QA tasks and the
# three multi-key needle tasks. Known properties worth keeping in mind when reading
# results, since each one bears on whether a drop with length is positional:
#
#   qa_1, qa_2     SQuAD / HotpotQA. Distractor paragraphs fill the context, so a longer
#                  context means more distractors, not just more distance. They draw from
#                  a FINITE question pool, so train and eval are kept on disjoint question
#                  ranges -- see QA_HOLDOUT. HotpotQA carries all ten of its paragraphs,
#                  which cannot be split, so some questions exceed 2048 outright and are
#                  skipped (logged by qa.py).
#   niah_multikey_1  essay haystack with 4 key/value needles; the query must pick one.
#                  Needle count is fixed, but the essay is always the corpus prefix, so
#                  eval at 8192 contains text never seen at the 2048 training length.
#   niah_multikey_2/3  "needle" haystack: the filler is itself key/value needles, so
#                  distractor count grows with length. _3 uses uuid keys and values, so
#                  it also needs long exact copies.
if $SANITY; then
    # Applied here, not in the sanity block above, because this line would otherwise
    # overwrite it -- config_tasks.sh is sourced after the grid is configured.
    TASKS=("${SANITY_TASKS[@]}")
else
    TASKS=("qa_1" "qa_2" "niah_multikey_1" "niah_multikey_2" "niah_multikey_3")
fi

# Fail here rather than partway through a submission: prepare.py looks each task up in
# synthetic.yaml, and a typo would otherwise surface one task into data generation.
for _t in "${TASKS[@]}"; do
    if ! grep -qE "^${_t}:" "${SCRIPT_DIR}/synthetic.yaml"; then
        echo "ERROR: task '${_t}' is not defined in ${SCRIPT_DIR}/synthetic.yaml." >&2
        exit 1
    fi
done
echo "Tasks: ${#TASKS[@]} (${TASKS[*]})"

# ====================== PATHS ================================================
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/../experiments}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
TRAIN_SCRIPT="${SCRIPT_DIR}/tmodel/train.py"

# ====================== SUMMARY MODE =========================================
if $SUMMARY; then
    # `nulls` is carried through because the metric is a plain substring match: an
    # empty prediction and a confidently wrong one both score 0.0, and only the null
    # count separates "the model emitted nothing" from "the model emitted the wrong
    # thing". Those two have completely different causes.
    echo "pe_type,encoder_mask,seq_length,task,score,nulls"
    for dir in "${EXP_ROOT}"/results/pe_*/synthetic/*/pred; do
        [ -f "${dir}/summary.csv" ] || continue
        # Parse path: .../pe_<PE>_enc<SPEC>/synthetic/<SEQ>/pred/summary.csv
        # Shell parameter expansion only -- `grep -oP` is GNU-specific and is not
        # available in the BSD grep shipped with macOS.
        seq_dir="${dir%/pred}"          # .../synthetic/<SEQ>
        seq="${seq_dir##*/}"            # <SEQ>
        cfg_dir="${seq_dir%/*}"         # .../synthetic
        cfg_dir="${cfg_dir%/*}"         # .../pe_<PE>_enc<SPEC>
        cfg="${cfg_dir##*/}"            # pe_<PE>_enc<SPEC>
        rest="${cfg#pe_}"               # <PE>_enc<SPEC>
        pe="${rest%_enc*}"              # <PE>
        enc="${rest##*_enc}"            # <SPEC>
        # summary.csv is a transposed frame written by eval/evaluate.py, so its first
        # line is pandas' integer column header and the task names are on line 2:
        #   0,1,2,...
        #   Tasks,<task>,...
        #   Score,<score>,...
        #   Nulls,<n>/<total>,...
        python3 -c "
import csv, sys
with open('${dir}/summary.csv') as f:
    rows = list(csv.reader(f))
by_label = {r[0]: r[1:] for r in rows if r}
tasks  = by_label.get('Tasks', [])
scores = by_label.get('Score', [])
nulls  = by_label.get('Nulls', [])
if not tasks:
    sys.exit(f'malformed summary: ${dir}/summary.csv')
nulls += [''] * (len(tasks) - len(nulls))
for t, s, n in zip(tasks, scores, nulls):
    print(f'${pe},${enc},${seq},{t},{s},{n}')
"
    done
    exit 0
fi

mkdir -p "${LOG_DIR}"

# ====================== ENVIRONMENT PREFLIGHT ================================
# Report every missing package at once. Without this a missing dependency fails one
# job at a time: submit, queue, fail, install one package, resubmit, repeat.
# Activation is done by hand rather than by sourcing bin/activate, because that script
# touches unset variables and this runs under 'set -u'. Putting the venv's bin first on
# PATH also makes a bare `python` resolve to it, which data/prepare.py depends on when
# it shells out to the generators.
setup_env() {
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        echo "--- Creating virtualenv: ${VENV_DIR} ---"
        if ! "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"; then
            echo "ERROR: could not create a virtualenv with '${BOOTSTRAP_PYTHON} -m venv'." >&2
            echo "       Point BOOTSTRAP_PYTHON at a usable interpreter, or load a python" >&2
            echo "       module first. Some sites need the python3-venv package." >&2
            exit 1
        fi
    fi

    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    unset PYTHONHOME 2>/dev/null || true
    echo "--- Using virtualenv: ${VENV_DIR} ---"

    # Reinstall when requirements.txt is newer than the last successful install, so
    # editing it is enough to refresh the environment. The marker is written only on
    # success, so an interrupted install is retried rather than assumed complete.
    local marker="${VENV_DIR}/.requirements-installed"
    if [ -f "${marker}" ] && [ ! "${REQUIREMENTS}" -nt "${marker}" ]; then
        return 0
    fi

    if [ ! -f "${REQUIREMENTS}" ]; then
        echo "ERROR: requirements file not found: ${REQUIREMENTS}" >&2
        exit 1
    fi

    # A CUDA-specific torch must go in first: installing it before requirements.txt
    # means the later 'torch>=2.1' line is already satisfied and will not pull a
    # different wheel over the top of it.
    if [ -n "${TORCH_SPEC}" ] && ! python -c "import torch" 2>/dev/null; then
        echo "--- Installing torch from TORCH_SPEC: ${TORCH_SPEC} ---"
        # shellcheck disable=SC2086
        _pip_install ${TORCH_SPEC}
    fi

    echo "--- Installing from ${REQUIREMENTS} ---"
    python -m pip install --quiet --upgrade pip || true
    _pip_install -r "${REQUIREMENTS}"
    touch "${marker}"
    echo "    environment ready"
}

# Reports missing packages as a space-separated list of pip names on stdout.
_missing_packages() {
    python - <<'PYCHECK'
import importlib.util

# python module name -> pip distribution name (they differ for some)
REQUIRED = {
    "torch": "torch", "transformers": "transformers", "numpy": "numpy",
    "scipy": "scipy", "nltk": "nltk", "wonderwords": "wonderwords",
    "tenacity": "tenacity", "pandas": "pandas", "yaml": "pyyaml",
    "tqdm": "tqdm", "requests": "requests",
}
print(" ".join(sorted({pip for mod, pip in REQUIRED.items()
                       if importlib.util.find_spec(mod) is None})))
PYCHECK
}

# Install the named pip packages into the *same* interpreter the checks use, then
# verify they import. `python -m pip` rather than a bare `pip`, which on a cluster
# often resolves to a different environment than `python` does.
_pip_install() {
    if ! $AUTO_INSTALL; then
        echo "ERROR: missing packages and AUTO_INSTALL=false." >&2
        echo "       Install them yourself:  python -m pip install $*" >&2
        exit 1
    fi
    echo "--- Installing: $* ---"
    if ! python -m pip install --no-input --disable-pip-version-check ${PIP_ARGS} "$@"; then
        echo "ERROR: pip install failed." >&2
        echo "       The usual cause is that this compute node has no outbound network." >&2
        echo "       Check with:" >&2
        echo "           srun --partition=${PARTITION} --time=00:02:00 python -m pip download --dest /tmp tqdm" >&2
        echo "       If that is the problem, point PIP_ARGS at a reachable source, e.g." >&2
        echo "           PIP_ARGS='--index-url https://<internal-mirror>/simple' bash run_experiments.sh" >&2
        echo "           PIP_ARGS='--no-index --find-links \$HOME/wheels' bash run_experiments.sh" >&2
        exit 1
    fi
}

check_environment() {
    echo "--- Checking Python environment ---"
    if ! python -c "import sys; print('    interpreter:', sys.executable)"; then
        echo "ERROR: no 'python' on PATH. data/prepare.py shells out to a bare 'python'," >&2
        echo "       so it must exist, not just 'python3'." >&2
        exit 1
    fi

    local missing
    missing="$(_missing_packages)"
    if [ -z "${missing}" ]; then
        echo "    all required packages present"
        return 0
    fi
    echo "    missing: ${missing}"

    # torch is deliberately never auto-installed. A plain PyPI wheel may be CPU-only
    # or built against the wrong CUDA, and the failure mode is silent: training falls
    # back to CPU and the sweep takes weeks instead of hours.
    case " ${missing} " in
        *" torch "*)
            if [ -z "${TORCH_SPEC}" ]; then
                echo "ERROR: torch is missing, and it is never installed from a bare" >&2
                echo "       'pip install torch'. A plain PyPI wheel may be CPU-only or built" >&2
                echo "       against the wrong CUDA, and that fails silently: training falls" >&2
                echo "       back to CPU and the sweep takes weeks instead of hours." >&2
                echo "       Name the wheel this cluster needs and the job will install it:" >&2
                echo "           TORCH_SPEC='torch --index-url https://download.pytorch.org/whl/cu121' bash run_experiments.sh" >&2
                exit 1
            fi
            echo "--- Installing torch from TORCH_SPEC: ${TORCH_SPEC} ---"
            # shellcheck disable=SC2086
            _pip_install ${TORCH_SPEC}
            missing="$(_missing_packages)"
            if [ -z "${missing}" ]; then
                echo "    environment ready"
                return 0
            fi
            echo "    still missing: ${missing}"
            ;;
    esac

    # shellcheck disable=SC2086
    _pip_install ${missing}

    missing="$(_missing_packages)"
    if [ -n "${missing}" ]; then
        echo "ERROR: still missing after install: ${missing}" >&2
        exit 1
    fi
    echo "    environment ready"
}

# Confirm torch can actually run a kernel on this node's GPU. A wheel that lacks
# compiled kernels for the device's compute capability imports fine, reports CUDA as
# available, and only fails on the first real operation -- as
# 'CUDA error: no kernel image is available for execution on the device', thrown
# somewhere deep in the model rather than at startup. Cheaper to find out here.
check_gpu() {
    python - <<'PYGPU'
import sys
import torch

if not torch.cuda.is_available():
    print("    no CUDA device visible (fine for CPU-only steps such as data prep)")
    sys.exit(0)

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"    gpu: {name} (sm_{major}{minor})")
print(f"    torch {torch.__version__} built for: {' '.join(archs)}")

try:
    # The smallest operation that needs a real kernel. This is what fails when the
    # wheel and the device disagree.
    torch.zeros(8, device="cuda").add_(1.0).sum().item()
except Exception as exc:
    sys.exit(
        f"ERROR: torch cannot run on this GPU.\n"
        f"       device : {name} (sm_{major}{minor})\n"
        f"       wheel  : torch {torch.__version__} built for {' '.join(archs)}\n"
        f"       cause  : {type(exc).__name__}: {exc}\n"
        f"       This wheel has no kernels for sm_{major}{minor}. Either pin the job to a\n"
        f"       GPU type the wheel supports (see GPU_SPEC in run_experiments.sh), or\n"
        f"       install a matching build, e.g.\n"
        f"           TORCH_SPEC='torch --index-url https://download.pytorch.org/whl/cu121'"
    )
print("    gpu check passed")
PYGPU
}

# ====================== FETCH SOURCE CORPORA =================================
# The needle tasks read Paul Graham essays and the QA tasks read SQuAD/HotpotQA.
# None of the three ships with the repository, and without them data generation
# fails on a missing file. Idempotent: existing files are left alone.
CORPUS_DIR="${SCRIPT_DIR}/data/synthetic/json"

fetch_corpora() {
    echo "--- Checking source corpora in ${CORPUS_DIR} ---"

    # Only fetch what the selected TASKS actually read. Most needle tasks use the noise
    # or needle haystacks and need no corpus at all, so a run restricted to those --
    # --sanity in particular -- should not pull down three datasets it will never open.
    # Which tasks use the essay haystack is set in synthetic.yaml (type_haystack: essay).
    # Must track every task with `type_haystack: essay` in synthetic.yaml. niah_single_3
    # was missing here, which went unnoticed only because niah_single_2 was always
    # selected alongside it; on a task list with single_3 but not single_2 the corpus
    # was never fetched and generation failed on the missing file.
    local essay_tasks=" niah_single_2 niah_single_3 niah_multikey_1 niah_multivalue niah_multiquery "
    local want_essay=false want_qa=false
    for t in "${TASKS[@]}"; do
        case "${essay_tasks}" in *" ${t} "*) want_essay=true ;; esac
        case "${t}" in qa_*) want_qa=true ;; esac
    done

    if ! $want_essay && ! $want_qa; then
        echo "    selected tasks need no external corpus, skipping"
        return 0
    fi

    local need_essay=false need_qa=false
    if $want_essay; then
        [ -f "${CORPUS_DIR}/PaulGrahamEssays.json" ] || need_essay=true
    fi
    if $want_qa; then
        { [ -f "${CORPUS_DIR}/squad.json" ] && [ -f "${CORPUS_DIR}/hotpotqa.json" ]; } || need_qa=true
    fi

    if ! $need_essay && ! $need_qa; then
        echo "    all corpora present, skipping download"
        return 0
    fi

    # These two are needed only by the essay downloader, so they are handled here
    # rather than in check_environment: a machine that already has the corpora
    # should not be made to install a downloader it will never run.
    if $need_essay && ! python -c "import html2text, bs4" 2>/dev/null; then
        _pip_install html2text beautifulsoup4
        python -c "import html2text, bs4" || {
            echo "ERROR: html2text/beautifulsoup4 still unavailable after install." >&2
            exit 1
        }
    fi

    ( cd "${CORPUS_DIR}" || exit 1
      if $need_essay; then
          echo "    downloading Paul Graham essays ..."
          python download_paulgraham_essay.py
      fi
      if $need_qa; then
          echo "    downloading SQuAD and HotpotQA ..."
          bash download_qa_dataset.sh
      fi
    )

    # Verify only what was actually required, for the same reason.
    local required=""
    $want_essay && required="PaulGrahamEssays.json"
    $want_qa && required="${required} squad.json hotpotqa.json"
    for f in ${required}; do
        if [ ! -f "${CORPUS_DIR}/${f}" ]; then
            echo "ERROR: ${CORPUS_DIR}/${f} is still missing after download." >&2
            exit 1
        fi
    done
    echo "    corpora ready"
}

# ====================== GENERATE SHARED TRAINING DATA ========================
# Training data is the same for all configs — generate once with seed=0.
TRAIN_DATA_DIR="${EXP_ROOT}/train_data"
# Eval data depends only on (length, task, seed) and the seed is fixed, so all 15
# cells would otherwise regenerate byte-identical files. Generate once and share.
EVAL_DATA_ROOT="${EXP_ROOT}/eval_data"

generate_eval_data() {
    echo "--- Generating RULER eval data (seed=${EVAL_SEED}) ---"
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local DATA_DIR="${EVAL_DATA_ROOT}/${SEQ_LEN}/data"
        mkdir -p "${DATA_DIR}"
        for TASK in "${TASKS[@]}"; do
            python "${SCRIPT_DIR}/data/prepare.py" \
                --save_dir   "${DATA_DIR}" \
                --benchmark  synthetic \
                --task       "${TASK}" \
                --tokenizer_path "${TOKENIZER}" \
                --tokenizer_type hf \
                --max_seq_length "${SEQ_LEN}" \
                --model_template_type base \
                --num_samples "${EVAL_SAMPLES}" \
                --random_seed "${EVAL_SEED}" \
                --qa_start 0 --qa_end "${QA_HOLDOUT}"
        done
    done
}

generate_train_data() {
    echo "--- Generating RULER training data (seed=${TRAIN_SEED}) ---"
    for SEQ_LEN in "${TRAIN_SEQ_LENGTHS[@]}"; do
        DATA_DIR="${TRAIN_DATA_DIR}/${SEQ_LEN}/data"
        mkdir -p "${DATA_DIR}"
        for TASK in "${TASKS[@]}"; do
            python "${SCRIPT_DIR}/data/prepare.py" \
                --save_dir   "${DATA_DIR}" \
                --benchmark  synthetic \
                --task       "${TASK}" \
                --tokenizer_path "${TOKENIZER}" \
                --tokenizer_type hf \
                --max_seq_length "${SEQ_LEN}" \
                --model_template_type base \
                --num_samples "${TRAIN_SAMPLES}" \
                --random_seed "${TRAIN_SEED}" \
                --qa_start "${QA_HOLDOUT}"
        done
    done
}

# Pass/fail gate for --sanity. Reads the summary evaluate.py just wrote and compares
# the score against SANITY_MIN_SCORE, exiting non-zero below it so the failure is
# visible in the job's exit status rather than only in a log nobody reads.
assert_sanity_score() {
    local summary="$1"
    if [ ! -f "${summary}" ]; then
        echo "SANITY FAILED: no summary at ${summary}" >&2
        exit 1
    fi
    python - "${summary}" "${SANITY_MIN_SCORE}" <<'PYSANITY'
import csv, sys

path, threshold = sys.argv[1], float(sys.argv[2])
rows = [r for r in csv.reader(open(path)) if r]
by_label = {r[0]: r[1:] for r in rows}
tasks = by_label.get("Tasks", [])
scores = by_label.get("Score", [])
nulls = by_label.get("Nulls", [])
if not tasks:
    sys.exit(f"SANITY FAILED: malformed summary {path}")

worst = None
for i, task in enumerate(tasks):
    score = float(scores[i])
    null = nulls[i] if i < len(nulls) else "?"
    print(f"    {task}: score={score} nulls={null}")
    if worst is None or score < worst:
        worst = score

if worst < threshold:
    sys.exit(
        f"\nSANITY FAILED: score {worst} is below the {threshold} threshold.\n"
        f"  A model cannot retrieve a needle from a context length it trained on.\n"
        f"  Something in the train/predict path is broken; the grid would produce\n"
        f"  zeros that look like a scientific result. Do not submit it.\n"
        f"  Check, in order: the copy rate logged by RulerDataset, the initial loss\n"
        f"  against ln(vocab_size), and whether val loss fell at all."
    )
print(f"\n    SANITY PASSED: {worst} >= {threshold}")
PYSANITY
}

# Longest answer in a task's eval set, in tokens, plus a small margin. Used to cap
# generation so a model cannot hedge -- emit several candidate orderings and let the
# substring metric credit one of them. The margin covers EOS and one stray token; it is
# deliberately too small to fit a second candidate answer.
#
# Mirrors how RulerDataset builds the training target: " " + " ".join(outputs).
answer_token_cap() {
    local jsonl="$1"
    python - "${jsonl}" "${TOKENIZER}" <<'PYCAP'
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2])
rows = [json.loads(l) for l in open(sys.argv[1])]
longest = max(
    len(tok.encode(" " + " ".join(str(o) for o in r.get("outputs", []) if str(o))))
    for r in rows
)
# Headroom, not +2. The cap was tight enough that a model whose output tokenises
# slightly differently from the gold string ran out of budget mid-answer and scored 0
# for a reason unrelated to retrieval -- visible as truncated 4-group uuids.
print(longest + 16)
PYCAP
}

# ====================== PER-EXPERIMENT JOB ====================================
run_experiment() {
    local pe="$1" enc_mask="$2"
    local EXP_NAME="pe_${pe}_enc${enc_mask}"
    local EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    local CKPT="${EXP_DIR}/best.pt"

    echo "=== ${EXP_NAME} ==="

    # ---- Phase 1: Train --------------------------------------------------
    # Skip training when a usable checkpoint is already there. TRAINING_COMPLETE.json is
    # written only after the epoch loop finishes, so it distinguishes "training ran to
    # completion" from "a checkpoint exists" -- best.pt is rewritten at every improvement,
    # so a job killed mid-training leaves one behind too, and reusing that would silently
    # evaluate an undertrained model.
    if ! $RETRAIN && [ -f "${CKPT}" ]; then
        if [ -f "${EXP_DIR}/TRAINING_COMPLETE.json" ]; then
            echo "--- Reusing checkpoint (training already completed) ---"
            python -c "
import json; m = json.load(open('${EXP_DIR}/TRAINING_COMPLETE.json'))
print(f\"    best val_loss {m['best_val_loss']:.4f} at epoch {m['best_epoch']}/{m['epoch_cap']}\")
print(f\"    stopped because: {m['stop_reason']}\")"
        else
            echo "WARNING: ${CKPT} exists but there is no TRAINING_COMPLETE.json, so this" >&2
            echo "         checkpoint may be from a run that was killed mid-training." >&2
            echo "         Reusing it anyway; pass RETRAIN=true to train from scratch." >&2
            python -c "
import torch; c = torch.load('${CKPT}', map_location='cpu', weights_only=False)
print(f\"    checkpoint is from epoch {c.get('epoch','?')}, val_loss {c.get('val_loss',float('nan')):.4f}\")"
        fi
    else
    python "${TRAIN_SCRIPT}" \
        --data_format  ruler \
        --data_dir     "${TRAIN_DATA_DIR}" \
        --pe_type      "${pe}" \
        --encoder_mask "${enc_mask}" \
        --d_model      "${D_MODEL}" \
        --num_heads    "${NUM_HEADS}" \
        --num_layers   "${NUM_LAYERS}" \
        --d_ff         "${D_FF}" \
        --dropout      "${DROPOUT}" \
        --max_len      "${MAX_LEN}" \
        --tokenizer    "${TOKENIZER}" \
        --src_len      "${SRC_LEN}" \
        --tgt_len      "${TGT_LEN}" \
        --epochs       "${EPOCHS}" \
        --batch_size   "${BATCH_SIZE}" \
        --grad_accum   "${GRAD_ACCUM}" \
        --early_stop_patience   "${EARLY_STOP_PATIENCE}" \
        --early_stop_min_delta  "${EARLY_STOP_MIN_DELTA}" \
        --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}" \
        --lr           "${LR}" \
        --warmup_steps "${WARMUP}" \
        --min_lr_frac  "${MIN_LR_FRAC}" \
        --seed         "${SEED}" \
        --fp16 \
        --output_dir   "${EXP_DIR}"
    fi

    # ---- Phase 2: Evaluate at each RULER sequence length ------------------
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local RESULTS="${EXP_ROOT}/results/${EXP_NAME}/synthetic/${SEQ_LEN}"
        # Read the shared eval data rather than regenerating it per configuration.
        local EVAL_DATA="${EVAL_DATA_ROOT}/${SEQ_LEN}/data"
        local PRED_DIR="${RESULTS}/pred"
        mkdir -p "${PRED_DIR}"

        for TASK in "${TASKS[@]}"; do
            # predict
            local CAP
            CAP="$(answer_token_cap "${EVAL_DATA}/${TASK}/validation.jsonl")"
            python "${SCRIPT_DIR}/pred/call_api.py" \
                --max_new_tokens "${CAP}" \
                --data_dir  "${EVAL_DATA}" \
                --save_dir  "${PRED_DIR}" \
                --benchmark synthetic \
                --task      "${TASK}" \
                --server_type      tmodel \
                --model_name_or_path "${CKPT}" \
                --temperature 0.0 \
                --top_k 1 \
                --top_p 1.0 \
                --batch_size 1
        done

        # evaluate
        python "${SCRIPT_DIR}/eval/evaluate.py" \
            --data_dir  "${PRED_DIR}" \
            --benchmark synthetic
    done

    # In sanity mode this is a pass/fail gate, not a measurement.
    if $SANITY; then
        assert_sanity_score "${EXP_ROOT}/results/${EXP_NAME}/synthetic/${EVAL_SEQ_LENGTHS[0]}/pred/summary.csv"
    fi

    # Every eval length is done and scored, so the weights have served their purpose:
    # predictions and summaries are on disk and are what the analysis reads. Deleting
    # here is safe because 'set -e' aborts before this line if any eval failed, so a
    # checkpoint is only removed once its results exist.
    if ! $KEEP_CHECKPOINTS; then
        rm -f "${CKPT}" "${EXP_DIR}/last.pt"
        echo "    removed checkpoint (KEEP_CHECKPOINTS=false); results are in ${EXP_ROOT}/results/${EXP_NAME}"
    fi

    echo "=== Done: ${EXP_NAME} ==="
}

# ====================== SHARED DATA PREP JOB =================================
# Training and eval data are identical across all 15 cells. Generating them inside
# every job would have 15 processes writing the same files concurrently, and
# prepare.py's "skip if the file exists" check is not atomic. Submit one prep job
# instead and make every experiment depend on it.
PREP_JOB_ID=""
if ! $LOCAL && ! $DRY_RUN; then
    PREP_JOB_ID=$(sbatch --parsable \
        --job-name="ruler_data_prep" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --output="${LOG_DIR}/data_prep_%j.out" \
        --error="${LOG_DIR}/data_prep_%j.err" \
        <<PREP_EOF
#!/bin/bash
set -euo pipefail
# Variables first: setup_env reads VENV_DIR and friends, and under 'set -u'
# referencing them before they are declared aborts the job immediately.
$(declare -p AUTO_INSTALL VENV_DIR REQUIREMENTS BOOTSTRAP_PYTHON PIP_ARGS TORCH_SPEC SCRIPT_DIR CORPUS_DIR TOKENIZER TASKS \
             TRAIN_DATA_DIR TRAIN_SEQ_LENGTHS TRAIN_SAMPLES TRAIN_SEED \
             EVAL_DATA_ROOT EVAL_SEQ_LENGTHS EVAL_SAMPLES EVAL_SEED QA_HOLDOUT)
$(declare -f setup_env)
$(declare -f _missing_packages)
$(declare -f _pip_install)
$(declare -f check_environment)
$(declare -f fetch_corpora)
$(declare -f generate_train_data)
$(declare -f generate_eval_data)
setup_env
check_environment
fetch_corpora
generate_train_data
generate_eval_data
PREP_EOF
    )
    echo "  -> submitted data prep job ${PREP_JOB_ID}"
fi

# ====================== DISPATCH LOOP ========================================
n_jobs=0

for cell in "${EXPERIMENTS[@]}"; do
    read -r pe enc_mask <<< "${cell}"
    EXP_NAME="pe_${pe}_enc${enc_mask}"

    if $LOCAL; then
        # ---------- local: prepare shared data once, then run each experiment ---
        if [ $n_jobs -eq 0 ]; then
            setup_env
            check_environment
            fetch_corpora
            generate_train_data
            generate_eval_data
        fi
        run_experiment "${pe}" "${enc_mask}"
        n_jobs=$((n_jobs + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[dry-run] would submit: ${EXP_NAME}"
        n_jobs=$((n_jobs + 1))
        continue
    fi

    # ---------- SLURM submission -----------------------------------------
    sbatch \
        --job-name="${EXP_NAME}" \
        --partition="${PARTITION}" \
        --gpus="${GPU_SPEC}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --dependency="afterok:${PREP_JOB_ID}" \
        --output="${LOG_DIR}/${EXP_NAME}_%j.out" \
        --error="${LOG_DIR}/${EXP_NAME}_%j.err" \
            <<SLURM_EOF
#!/bin/bash
set -euo pipefail
# Variables first: setup_env reads VENV_DIR and friends, and under 'set -u'
# referencing them before they are declared aborts the job immediately.
# NOTE: no 2>/dev/null on declare -p. Silently dropping an unset variable here would
# surface much later as an empty path or a skipped flag inside the job.
#
# Experiment jobs may install too. Normally they never need to: they depend on the
# prep job via afterok, so it has already installed into the shared environment by
# the time any of these start, and check_environment finds nothing missing. This
# matters only when the environment is not shared across nodes, where the prep job's
# install would not be visible here and hardcoding false would strand every job with
# no way to recover.
$(declare -p AUTO_INSTALL VENV_DIR REQUIREMENTS BOOTSTRAP_PYTHON PIP_ARGS TORCH_SPEC)
$(declare -p KEEP_CHECKPOINTS RETRAIN SANITY SANITY_MIN_SCORE)
$(declare -p EXP_ROOT TRAIN_DATA_DIR EVAL_DATA_ROOT TRAIN_SCRIPT D_MODEL NUM_HEADS \
             NUM_LAYERS D_FF DROPOUT MAX_LEN TOKENIZER SRC_LEN TGT_LEN EPOCHS \
             BATCH_SIZE GRAD_ACCUM LR WARMUP SEED EVAL_SEQ_LENGTHS EVAL_SAMPLES \
             EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA EARLY_STOP_MIN_EPOCHS MIN_LR_FRAC \
             EVAL_SEED QA_HOLDOUT TASKS SCRIPT_DIR)
$(declare -f setup_env)
$(declare -f _missing_packages)
$(declare -f _pip_install)
$(declare -f check_environment)
$(declare -f check_gpu)
$(declare -f assert_sanity_score)
$(declare -f answer_token_cap)
$(declare -f run_experiment)

setup_env

echo "Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# Data is produced by the prep job this one depends on; nothing to generate here.
check_environment
check_gpu
run_experiment "${pe}" "${enc_mask}"
SLURM_EOF

    echo "  -> submitted ${EXP_NAME}"
    n_jobs=$((n_jobs + 1))
done

echo ""
echo "========================================================"
echo "  ${n_jobs} experiments dispatched"
echo "  Results   : ${EXP_ROOT}/results/"
echo "  Summaries : bash $0 --summary"
echo ""
echo "  Monitor   : squeue -u \$USER"
echo "  Logs      : ${LOG_DIR}/"
echo "========================================================"
