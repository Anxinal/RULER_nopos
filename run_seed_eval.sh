#!/bin/bash
# =============================================================================
# run_seed_eval.sh
#
# Evaluation only. For every seed in SEEDS, finds the runs that already have a trained
# checkpoint (<EXP_ROOT>/<run>_s<seed>/best.pt) and predicts and scores them. Nothing is
# trained and no data is generated: a seed with no checkpoint is reported and skipped.
#
# Use it to finish a run whose evaluation was cut short, or to score existing
# checkpoints again. It stands alone -- it does not call run_seed_experiments.sh -- so
# EVAL_TASKS and EVAL_SEQ_LENGTHS below must name data that script has already
# generated under <EXP_ROOT>/eval_data.
#
#   bash run_seed_eval.sh              # submit one Slurm job per run found
#   bash run_seed_eval.sh --dry-run    # list the runs that would be evaluated
#   bash run_seed_eval.sh --local      # evaluate them here, one after another
#   SEEDS="96" PRED_SAMPLES=300 bash run_seed_eval.sh
#
# Prediction runs in chunks of PRED_CHUNK samples, each in a fresh process. A job was
# OOM-killed on HOST memory (64G) at sample 960 of one 1000-sample prediction pass, so
# memory grows with the samples a process has handled; a new process per chunk bounds
# it whatever the cause. call_api.py prints its peak host memory at the end of each
# chunk. Predictions already on disk are kept, so a run resumes where it stopped.
#
# Scores land where the sweep puts them; collect them with
#   bash run_seed_experiments.sh --summary
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${REPO_DIR}/scripts"

# ====================== SLURM ================================================
PARTITION="${PARTITION:-gpu}"
GPU_SPEC="${GPU_SPEC:-h100-96:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-05:00:00}"

# ====================== PATHS ================================================
# Same defaults as run_seed_experiments.sh, so the two find the same files.
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
EXP_ROOT="${EXP_ROOT:-${REPO_DIR}/experiments}"
EVAL_DATA_ROOT="${EXP_ROOT}/eval_data"
LOG_DIR="${EXP_ROOT}/slurm_logs"
TOKENIZER="${TOKENIZER:-gpt2}"

# ====================== WHAT TO EVALUATE =====================================
read -r -a SEEDS <<< "${SEEDS:-96 97}"
# Keep in step with run_seed_experiments.sh: these must already be generated.
read -r -a EVAL_SEQ_LENGTHS <<< "${EVAL_SEQ_LENGTHS:-2048 4096 8192}"
read -r -a EVAL_TASKS <<< "${EVAL_TASKS:-vt_2chain vt_4chain niah_multikey_1 niah_multikey_2 niah_multikey_3}"

# Samples predicted and scored per task and length: the first PRED_SAMPLES of each eval
# file (the files hold 1000). PRED_CHUNK is how many one process handles before the
# next one takes over.
PRED_SAMPLES="${PRED_SAMPLES:-500}"
PRED_CHUNK="${PRED_CHUNK:-100}"

# Checkpoints are kept, as in the sweep, since a second pass (more samples, another
# length) needs them. Set false to delete a run's checkpoints once it is scored.
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-true}"

# ====================== FLAGS ================================================
DRY_RUN=false
LOCAL=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --local)   LOCAL=true ;;
        *) echo "ERROR: unknown argument '$arg' (expected --dry-run or --local)." >&2
           exit 1 ;;
    esac
done

# ====================== HELPERS ==============================================
# Activated by hand rather than by sourcing bin/activate, which touches unset variables
# and so fails under 'set -u'. The environment is the one training built; nothing is
# installed here.
activate_env() {
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        echo "ERROR: no virtualenv at ${VENV_DIR}. It is created by" >&2
        echo "       run_seed_experiments.sh; set VENV_DIR if it lives elsewhere." >&2
        exit 1
    fi
    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    unset PYTHONHOME 2>/dev/null || true
}

# Longest answer in a task's eval set, in tokens, plus headroom. Caps generation so a
# model cannot emit several candidate answers and let the substring metric credit one.
# Same computation as answer_token_cap in run_seed_experiments.sh.
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
print(longest + 16)
PYCAP
}

# Predict and score one run at every eval length.
eval_run() {
    local EXP_NAME="$1"
    local EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    local CKPT="${EXP_DIR}/best.pt"

    echo "=== ${EXP_NAME} ==="
    if [ ! -f "${EXP_DIR}/TRAINING_COMPLETE.json" ]; then
        echo "WARNING: ${EXP_DIR} has no TRAINING_COMPLETE.json, so best.pt may be from a" >&2
        echo "         run that was killed mid-training. Evaluating it anyway." >&2
    fi
    echo "    device: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY -- no CUDA device visible')")"

    local SEQ_LEN TASK
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local EVAL_DATA="${EVAL_DATA_ROOT}/${SEQ_LEN}/data"
        local PRED_DIR="${EXP_ROOT}/results/${EXP_NAME}/synthetic/${SEQ_LEN}/pred"
        mkdir -p "${PRED_DIR}"

        for TASK in "${EVAL_TASKS[@]}"; do
            local DATA_FILE="${EVAL_DATA}/${TASK}/validation.jsonl"
            if [ ! -f "${DATA_FILE}" ]; then
                echo "ERROR: no eval data at ${DATA_FILE}. Generate it with" >&2
                echo "       run_seed_experiments.sh, or fix EVAL_TASKS / EVAL_SEQ_LENGTHS." >&2
                return 1
            fi
            local CAP; CAP="$(answer_token_cap "${DATA_FILE}")"
            local PRED_FILE="${PRED_DIR}/${TASK}.jsonl"

            local upto=0 have
            while [ "${upto}" -lt "${PRED_SAMPLES}" ]; do
                upto=$(( upto + PRED_CHUNK ))
                if [ "${upto}" -gt "${PRED_SAMPLES}" ]; then upto="${PRED_SAMPLES}"; fi
                # Predictions are written in order, so this many lines means the first
                # `upto` samples are done and the process need not even load the model.
                have=0
                if [ -f "${PRED_FILE}" ]; then have="$(wc -l < "${PRED_FILE}" | tr -d ' ')"; fi
                if [ "${have}" -ge "${upto}" ]; then continue; fi
                echo "--- ${TASK} @ ${SEQ_LEN}: samples ${have}..${upto} of ${PRED_SAMPLES} ---"
                python "${SCRIPT_DIR}/pred/call_api.py" \
                    --max_new_tokens "${CAP}" \
                    --num_samples "${upto}" \
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
        done

        python "${SCRIPT_DIR}/eval/evaluate.py" \
            --data_dir  "${PRED_DIR}" \
            --benchmark synthetic
    done

    # Reached only if every length was predicted and scored: 'set -e' stops earlier.
    if ! $KEEP_CHECKPOINTS; then
        rm -f "${CKPT}" "${EXP_DIR}/last.pt" "${EXP_DIR}/starter.pt"
        echo "    removed checkpoint (KEEP_CHECKPOINTS=false)"
    fi
    echo "=== Done: ${EXP_NAME} ==="
}

# ====================== FIND THE RUNS ========================================
# A run is evaluated only if its seed is in SEEDS and its best.pt is present.
RUNS=()
for seed in "${SEEDS[@]}"; do
    found=0
    for ckpt in "${EXP_ROOT}"/*_s"${seed}"/best.pt; do
        [ -f "${ckpt}" ] || continue      # the unexpanded pattern, when nothing matches
        RUNS+=("$(basename "$(dirname "${ckpt}")")")
        found=$((found + 1))
    done
    if [ "${found}" -eq 0 ]; then
        echo "  [skip] seed ${seed}: no checkpoint (${EXP_ROOT}/*_s${seed}/best.pt)"
    fi
done

echo "Evaluating ${#RUNS[@]} run(s): ${PRED_SAMPLES} samples per task (chunks of" \
     "${PRED_CHUNK}), tasks ${EVAL_TASKS[*]}, lengths ${EVAL_SEQ_LENGTHS[*]}"
if [ "${#RUNS[@]}" -eq 0 ]; then
    exit 0
fi

# ====================== DISPATCH =============================================
mkdir -p "${LOG_DIR}"
for EXP_NAME in "${RUNS[@]}"; do
    if $DRY_RUN; then
        echo "[dry-run] ${EXP_NAME}"
        continue
    fi

    if $LOCAL; then
        activate_env
        eval_run "${EXP_NAME}"
        continue
    fi

    sbatch \
        --job-name="eval_${EXP_NAME}" \
        --partition="${PARTITION}" \
        --gpus="${GPU_SPEC}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --output="${LOG_DIR}/eval_${EXP_NAME}_%j.out" \
        --error="${LOG_DIR}/eval_${EXP_NAME}_%j.err" \
            <<SLURM_EOF
#!/bin/bash
set -euo pipefail
$(declare -p VENV_DIR EXP_ROOT EVAL_DATA_ROOT SCRIPT_DIR TOKENIZER EVAL_SEQ_LENGTHS \
             EVAL_TASKS PRED_SAMPLES PRED_CHUNK KEEP_CHECKPOINTS)
$(declare -f activate_env)
$(declare -f answer_token_cap)
$(declare -f eval_run)

activate_env
echo "Node: \$(hostname)"
eval_run "${EXP_NAME}"
SLURM_EOF
    echo "  -> submitted eval_${EXP_NAME}"
done
