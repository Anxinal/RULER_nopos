#!/bin/bash
# Train position probes on layers 2-5 of every vanilla-transformer run that has a
# checkpoint: transformer_mask with and without sinusoidal PE, every mask and seed.
#
#   bash run_position_probes.sh              # one Slurm job per run
#   bash run_position_probes.sh --local      # run here, one after another
#   bash run_position_probes.sh --dry-run    # list what would run
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
EXP_ROOT="${EXP_ROOT:-${REPO_DIR}/experiments}"
DATA_DIR="${DATA_DIR:-${EXP_ROOT}/train_data/2048/data}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
LAYERS="${LAYERS:-1 2 3 4}"
STACK="${STACK:-encoder}"

PARTITION="${PARTITION:-gpu}"
GPU_SPEC="${GPU_SPEC:-h100-96:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-06:00:00}"

MODE="${1:-slurm}"

# Probes every layer of one run; a layer whose probe file exists is skipped.
probe_run() {
    local run="$1" layer
    export PATH="${VENV_DIR}/bin:${PATH}"
    cd "${REPO_DIR}"
    for layer in ${LAYERS}; do
        if [ -f "${run}/position_probe_layer${layer}.pt" ]; then
            echo "--- $(basename "${run}") layer ${layer}: already done ---"
            continue
        fi
        echo "--- $(basename "${run}") layer ${layer} ---"
        python train_position_probe.py "${run}" "${layer}" --data_dir "${DATA_DIR}" --stack "${STACK}"
    done
}

mkdir -p "${LOG_DIR}"
for ckpt in "${EXP_ROOT}"/transformer_mask_pe*_s*/best.pt; do
    [ -f "${ckpt}" ] || { echo "no transformer_mask checkpoints under ${EXP_ROOT}"; exit 0; }
    run="$(dirname "${ckpt}")"
    name="$(basename "${run}")"
    case "${MODE}" in
        --dry-run) echo "[dry-run] ${name}: layers ${LAYERS}" ;;
        --local)   probe_run "${run}" ;;
        slurm)
            sbatch --job-name="probe_${name}" --partition="${PARTITION}" --gpus="${GPU_SPEC}" \
                   --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" \
                   --output="${LOG_DIR}/probe_${name}_%j.out" \
                   --error="${LOG_DIR}/probe_${name}_%j.err" <<EOF
#!/bin/bash
set -euo pipefail
$(declare -p REPO_DIR VENV_DIR DATA_DIR LAYERS STACK)
$(declare -f probe_run)
probe_run "${run}"
EOF
            echo "  -> submitted probe_${name}" ;;
        *) echo "ERROR: unknown argument '${MODE}' (expected --local or --dry-run)." >&2; exit 1 ;;
    esac
done
