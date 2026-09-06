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

# ====================== MODEL ================================================
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=8
D_FF=2048
DROPOUT=0.1
MAX_LEN=8192               # PE buffer length (must >= longest TRAIN seq length)
TOKENIZER="gpt2"

# ====================== DATA =================================================
TRAIN_SEQ_LENGTHS=(4096)    # seq lengths for training data
TRAIN_SAMPLES=500           # samples per task per seq length
TRAIN_SEED=0                # separate seed to avoid data leakage

EVAL_SEQ_LENGTHS=(4096 8192 16384 32768)
EVAL_SAMPLES=500
EVAL_SEED=42                # RULER default

# ====================== TRAINING =============================================
EPOCHS=10
BATCH_SIZE=8
LR=3e-4
WARMUP=1000
SRC_LEN=4096                # max encoder tokens during training
TGT_LEN=128                # max decoder tokens during training
SEED=42

# ====================== EXPERIMENT GRID ======================================
PE_TYPES=("none" "sinusoidal" "learned" "rope" "alibi")
#   "ENCODER_MASK DECODER_MASK"
MASK_CONFIGS=("B C" "C C" "F C")

# Task list (must match entries in synthetic.yaml)
source "${SCRIPT_DIR}/config_tasks.sh"
TASKS=("${synthetic[@]}")

# ====================== PATHS ================================================
EXP_ROOT="${SCRIPT_DIR}/../experiments"
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
    echo "pe_type,encoder_mask,decoder_mask,seq_length,task,score"
    for dir in "${EXP_ROOT}"/results/pe_*/synthetic/*/pred; do
        [ -f "${dir}/summary.csv" ] || continue
        # parse path: .../pe_X_encY_decZ/synthetic/SEQ/pred/summary.csv
        cfg=$(echo "$dir" | grep -oP 'pe_[^/]+')
        seq=$(echo "$dir" | grep -oP 'synthetic/\K[0-9]+')
        pe=$(echo "$cfg"  | sed 's/pe_\(.*\)_enc.*/\1/')
        enc=$(echo "$cfg" | sed 's/.*_enc\(.\)_.*/\1/')
        dec=$(echo "$cfg" | sed 's/.*_dec\(.\)/\1/')
        # read CSV columns (row 0 = task names, row 1 = scores)
        python3 -c "
import csv, sys
with open('${dir}/summary.csv') as f:
    rows = list(csv.reader(f))
tasks  = rows[0][1:]
scores = rows[1][1:]
for t, s in zip(tasks, scores):
    print(f'${pe},${enc},${dec},${seq},{t},{s}')
" 2>/dev/null
    done
    exit 0
fi

mkdir -p "${LOG_DIR}"

# ====================== GENERATE SHARED TRAINING DATA ========================
# Training data is the same for all configs — generate once with seed=0.
TRAIN_DATA_DIR="${EXP_ROOT}/train_data"

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
    local pe="$1" enc_mask="$2" dec_mask="$3"
    local EXP_NAME="pe_${pe}_enc${enc_mask}_dec${dec_mask}"
    local EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    local CKPT="${EXP_DIR}/best.pt"

    echo "=== ${EXP_NAME} ==="

    # ---- Phase 1: Train --------------------------------------------------
    python "${TRAIN_SCRIPT}" \
        --data_format  ruler \
        --data_dir     "${TRAIN_DATA_DIR}" \
        --pe_type      "${pe}" \
        --encoder_mask "${enc_mask}" \
        --decoder_mask "${dec_mask}" \
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
        --lr           "${LR}" \
        --warmup_steps "${WARMUP}" \
        --seed         "${SEED}" \
        --fp16 \
        --output_dir   "${EXP_DIR}"

    # ---- Phase 2: Evaluate at each RULER sequence length ------------------
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local RESULTS="${EXP_ROOT}/results/${EXP_NAME}/synthetic/${SEQ_LEN}"
        local EVAL_DATA="${RESULTS}/data"
        local PRED_DIR="${RESULTS}/pred"
        mkdir -p "${EVAL_DATA}" "${PRED_DIR}"

        for TASK in "${TASKS[@]}"; do
            # generate eval data (default seed=42)
            python "${SCRIPT_DIR}/data/prepare.py" \
                --save_dir   "${EVAL_DATA}" \
                --benchmark  synthetic \
                --task       "${TASK}" \
                --tokenizer_path "${TOKENIZER}" \
                --tokenizer_type hf \
                --max_seq_length "${SEQ_LEN}" \
                --model_template_type base \
                --num_samples "${EVAL_SAMPLES}" \
                --random_seed "${EVAL_SEED}"

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

# ====================== DISPATCH LOOP ========================================
n_jobs=0

for pe in "${PE_TYPES[@]}"; do
    for mask_cfg in "${MASK_CONFIGS[@]}"; do
        read -r enc_mask dec_mask <<< "${mask_cfg}"
        EXP_NAME="pe_${pe}_enc${enc_mask}_dec${dec_mask}"

        if $LOCAL; then
            # ---------- local: generate data once, then run each experiment ---
            if [ $n_jobs -eq 0 ]; then generate_train_data; fi
            run_experiment "${pe}" "${enc_mask}" "${dec_mask}"
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
            --output="${LOG_DIR}/${EXP_NAME}_%j.out" \
            --error="${LOG_DIR}/${EXP_NAME}_%j.err" \
            <<SLURM_EOF
#!/bin/bash
set -euo pipefail
if command -v conda &>/dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

echo "Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# Generate training data (idempotent — skips if files exist)
$(declare -f generate_train_data)
$(declare -p TRAIN_DATA_DIR TRAIN_SEQ_LENGTHS TASKS TRAIN_SAMPLES TRAIN_SEED SCRIPT_DIR TOKENIZER 2>/dev/null)
generate_train_data

# Run experiment
$(declare -f run_experiment)
$(declare -p EXP_ROOT TRAIN_DATA_DIR TRAIN_SCRIPT D_MODEL NUM_HEADS NUM_LAYERS D_FF DROPOUT MAX_LEN TOKENIZER SRC_LEN TGT_LEN EPOCHS BATCH_SIZE LR WARMUP SEED EVAL_SEQ_LENGTHS EVAL_SAMPLES EVAL_SEED TASKS SCRIPT_DIR 2>/dev/null)
run_experiment "${pe}" "${enc_mask}" "${dec_mask}"
SLURM_EOF

        echo "  -> submitted ${EXP_NAME}"
        n_jobs=$((n_jobs + 1))
    done
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
