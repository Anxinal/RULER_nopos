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
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-48:00:00}"
CONDA_ENV="${CONDA_ENV:-ruler}"
# Install missing Python packages from inside the job rather than by hand first.
# torch is always excluded; see check_environment for why.
AUTO_INSTALL="${AUTO_INSTALL:-true}"
# Extra flags for every pip install, for sites where PyPI is not directly reachable
# from a compute node. Examples:
#   PIP_ARGS='--index-url https://<internal-mirror>/simple'
#   PIP_ARGS='--no-index --find-links $HOME/wheels'
PIP_ARGS="${PIP_ARGS:-}"

# ====================== MODEL ================================================
D_MODEL=1024
NUM_HEADS=8
NUM_LAYERS=8
D_FF=2048
DROPOUT=0.2
MAX_LEN=16384     # PE buffer length (must be >= longest EVAL seq length)
TOKENIZER="gpt2"

# ====================== DATA =================================================
# Length-generalisation design: train at one length, evaluate at multiples of it.
# Every eval length must stay <= MAX_LEN. The sinusoidal, learned and rotary buffers
# are all sized by MAX_LEN and now raise a clear error rather than failing obscurely
# past the end of the table.
TRAIN_SEQ_LENGTHS=(2048)    # seq lengths for training data
TRAIN_SAMPLES=2000          # samples per task per seq length
TRAIN_SEED=0                # separate seed to avoid data leakage

EVAL_SEQ_LENGTHS=(4096 6144 8192)   # 2x, 3x, 4x the training length
EVAL_SAMPLES=500
EVAL_SEED=42                # RULER default

# ====================== TRAINING =============================================
EPOCHS=20
BATCH_SIZE=8
GRAD_ACCUM=8                # effective batch = BATCH_SIZE * GRAD_ACCUM
LR=1e-4
WARMUP=1000
SRC_LEN=2048                # max encoder tokens during training
TGT_LEN=128                # max decoder tokens during training
SEED=42

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
#   * "none" is crossed with every mask -- the mask is then the only thing that can
#     carry position, which is the arm the whole experiment exists to measure.
#   * "sinusoidal" is crossed with every mask as the vanilla-transformer reference.
#   * every other encoding runs only at NO_MASK_SPEC, isolating the encoding.
PE_CROSSED_WITH_MASKS=("none" "sinusoidal")
PE_NO_MASK_ONLY=("learned" "rope" "alibi")

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
for pe in "${PE_NO_MASK_ONLY[@]}"; do
    EXPERIMENTS+=("${pe} ${NO_MASK_SPEC}")
done

# Task list (must match entries in synthetic.yaml)
source "${SCRIPT_DIR}/config_tasks.sh"
TASKS=("${synthetic[@]}")

# ====================== PATHS ================================================
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/../experiments}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
TRAIN_SCRIPT="${SCRIPT_DIR}/tmodel/train.py"

# ====================== FLAGS ================================================
DRY_RUN=false
LOCAL=false
SUMMARY=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --local)   LOCAL=true ;;
        --summary) SUMMARY=true ;;
    esac
done

# ====================== SUMMARY MODE =========================================
if $SUMMARY; then
    echo "pe_type,encoder_mask,seq_length,task,score"
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
if not tasks:
    sys.exit(f'malformed summary: ${dir}/summary.csv')
for t, s in zip(tasks, scores):
    print(f'${pe},${enc},${seq},{t},{s}')
"
    done
    exit 0
fi

mkdir -p "${LOG_DIR}"

# ====================== ENVIRONMENT PREFLIGHT ================================
# Report every missing package at once. Without this a missing dependency fails one
# job at a time: submit, queue, fail, install one package, resubmit, repeat.
# NOTE: this deliberately runs `python`, not `python3`, because data/prepare.py
# shells out to a bare `python` -- so if that name is not on PATH, this is where it
# surfaces rather than thirteen lines deep in a captured subprocess stderr.
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
            echo "ERROR: torch is missing, and this script will not install it for you." >&2
            echo "       Install the wheel matching this cluster's CUDA, e.g." >&2
            echo "           pip install torch --index-url https://download.pytorch.org/whl/cu121" >&2
            exit 1
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

# ====================== FETCH SOURCE CORPORA =================================
# The needle tasks read Paul Graham essays and the QA tasks read SQuAD/HotpotQA.
# None of the three ships with the repository, and without them data generation
# fails on a missing file. Idempotent: existing files are left alone.
CORPUS_DIR="${SCRIPT_DIR}/data/synthetic/json"

fetch_corpora() {
    echo "--- Checking source corpora in ${CORPUS_DIR} ---"
    local need_essay=false need_qa=false
    [ -f "${CORPUS_DIR}/PaulGrahamEssays.json" ] || need_essay=true
    { [ -f "${CORPUS_DIR}/squad.json" ] && [ -f "${CORPUS_DIR}/hotpotqa.json" ]; } || need_qa=true

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

    for f in PaulGrahamEssays.json squad.json hotpotqa.json; do
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
                --random_seed "${EVAL_SEED}"
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
                --random_seed "${TRAIN_SEED}"
        done
    done
}

# ====================== PER-EXPERIMENT JOB ====================================
run_experiment() {
    local pe="$1" enc_mask="$2"
    local EXP_NAME="pe_${pe}_enc${enc_mask}"
    local EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    local CKPT="${EXP_DIR}/best.pt"

    echo "=== ${EXP_NAME} ==="

    # ---- Phase 1: Train --------------------------------------------------
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
        --lr           "${LR}" \
        --warmup_steps "${WARMUP}" \
        --seed         "${SEED}" \
        --fp16 \
        --output_dir   "${EXP_DIR}"

    # ---- Phase 2: Evaluate at each RULER sequence length ------------------
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local RESULTS="${EXP_ROOT}/results/${EXP_NAME}/synthetic/${SEQ_LEN}"
        # Read the shared eval data rather than regenerating it per configuration.
        local EVAL_DATA="${EVAL_DATA_ROOT}/${SEQ_LEN}/data"
        local PRED_DIR="${RESULTS}/pred"
        mkdir -p "${PRED_DIR}"

        for TASK in "${TASKS[@]}"; do
            # predict
            python "${SCRIPT_DIR}/pred/call_api.py" \
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
if command -v conda &>/dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
else
    echo "WARNING: conda not found; running in the ambient environment." >&2
fi
$(declare -f _missing_packages)
$(declare -f _pip_install)
$(declare -f check_environment)
$(declare -f fetch_corpora)
$(declare -f generate_train_data)
$(declare -f generate_eval_data)
$(declare -p AUTO_INSTALL CONDA_ENV PIP_ARGS SCRIPT_DIR CORPUS_DIR TOKENIZER TASKS \
             TRAIN_DATA_DIR TRAIN_SEQ_LENGTHS TRAIN_SAMPLES TRAIN_SEED \
             EVAL_DATA_ROOT EVAL_SEQ_LENGTHS EVAL_SAMPLES EVAL_SEED)
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
        --gres="gpu:${GPUS}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --dependency="afterok:${PREP_JOB_ID}" \
        --output="${LOG_DIR}/${EXP_NAME}_%j.out" \
        --error="${LOG_DIR}/${EXP_NAME}_%j.err" \
            <<SLURM_EOF
#!/bin/bash
set -euo pipefail
if command -v conda &>/dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
else
    echo "WARNING: conda not found; running in the ambient environment." >&2
fi

echo "Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# Data is produced by the prep job this one depends on; nothing to generate here.
# NOTE: no 2>/dev/null on declare -p. Silently dropping an unset variable here would
# surface much later as an empty path or a skipped flag inside the job.
$(declare -f _missing_packages)
$(declare -f _pip_install)
$(declare -f check_environment)
$(declare -f run_experiment)
# Experiment jobs may install too. Normally they never need to: they depend on the
# prep job via afterok, so it has already installed into the shared environment by
# the time any of these start, and check_environment finds nothing missing. This
# matters only when the environment is not shared across nodes, where the prep job's
# install would not be visible here and hardcoding false would strand every job with
# no way to recover.
$(declare -p AUTO_INSTALL CONDA_ENV PIP_ARGS)
$(declare -p EXP_ROOT TRAIN_DATA_DIR EVAL_DATA_ROOT TRAIN_SCRIPT D_MODEL NUM_HEADS \
             NUM_LAYERS D_FF DROPOUT MAX_LEN TOKENIZER SRC_LEN TGT_LEN EPOCHS \
             BATCH_SIZE GRAD_ACCUM LR WARMUP SEED EVAL_SEQ_LENGTHS EVAL_SAMPLES \
             EVAL_SEED TASKS SCRIPT_DIR)
check_environment
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
