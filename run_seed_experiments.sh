#!/bin/bash
# =============================================================================
# run_seed_experiments.sh
#
# Seed sweep on the RULER benchmark: every arm below is trained and evaluated with
# each of SEEDS, one Slurm job per (arm, seed).
#
# Arms (4), see TM_ARMS / OTHER_MODELS:
#   transformer_mask (maskedVanilla, encoder-decoder):
#       pe none,       mask CCCCFFFF
#       pe none,       mask CCCCCCCC
#       pe sinusoidal, mask BBBBBBBB   (vanilla transformer, no mask)
#   roformer  (decoder-only, RoPE)
# transformer_mask uses nn.Transformer's sizes, pre-norm; roformer is sized to match it.
# Both tie their input and output embeddings, ~70M parameters each
# (see tmodel/models.py).
#
# Pipeline (per arm and seed):
#   1. Generate RULER training data   (scripts/data/prepare.py, once, shared)
#   2. Train with wandb tracking       (train_wandb.py: starter stage on STARTER_TASKS,
#                                       then the full stage on TRAIN_TASKS)
#   3. Generate RULER eval data       (scripts/data/prepare.py, once, shared)
#   4. Run predictions                (scripts/pred/call_api.py --server_type tmodel)
#   5. Compute metrics                (scripts/eval/evaluate.py)
#
# Modes
# -----
#   bash run_seed_experiments.sh              # submit to SLURM
#   bash run_seed_experiments.sh --dry-run    # print each job's training command
#   bash run_seed_experiments.sh --local      # run sequentially, no SLURM
#   bash run_seed_experiments.sh --summary    # collect results into CSV
#
# To evaluate existing checkpoints without training, use run_seed_eval.sh.
#
# wandb: jobs need credentials on the compute nodes -- `wandb login` on a shared home
# (writes ~/.netrc) or WANDB_API_KEY exported when submitting (sbatch passes the
# environment through). Set WANDB_MODE=offline to log locally and `wandb sync` later.
# =============================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The pipeline scripts (data/, pred/, eval/, config_tasks.sh, synthetic.yaml) live in
# scripts/; every path below is relative to it, as in scripts/run_experiments.sh.
SCRIPT_DIR="${REPO_DIR}/scripts"

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
TIME="${TIME:-16:00:00}"
# Checkpoints are deleted once a cell has been evaluated at every length, since
# the predictions and summaries are what the analysis reads. At ~0.9 GB per cell
# this is the difference between ~8 GB and ~0 GB of standing disk. Set true to keep
# them, e.g. to re-run evaluation later without retraining.
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-false}"
# Reuse an existing checkpoint instead of retraining, so a cell whose evaluation was cut
# short (a TIME kill lands before the checkpoint s deleted) can be resubmitted and go
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
# No size settings: the sizes are fixed in tmodel/models.py (nn.Transformer's, with
# roformer matched to them). transformer_mask has 8 heads, so each
# mask spec below is one code per head. MAX_LEN is set after the data section.
TM_NUM_HEADS=8
TOKENIZER="gpt2"
# The one override of the library defaults: every dropout rate in every model (0.1 in
# all three libraries), attention dropout included. Passed as train.py --dropout.
DROPOUT=0.0

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
# Samples per task per seq length -- PER TASK, so the seven TRAIN_TASKS generate 224k
# samples (train.py holds 10% of them out for validation). For
# the QA tasks this exceeds the unique training questions (see QA_HOLDOUT), so each is
# reused a few times with a different draw of distractor paragraphs.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"   # samples per task per seq length
# MUST differ from EVAL_SEED. Both were 42, which made the two prepare.py runs produce
# identical RNG streams at the same --max_seq_length, so every eval sample at the train
# length was literally a training sample and that whole column measured memorisation.
TRAIN_SEED=934

# 2048 is the training length, then 2x and 4x it.
EVAL_SEQ_LENGTHS=(2048 4096 8192)
EVAL_SAMPLES=1000
# How many of those are actually predicted and scored: the first PRED_SAMPLES of each
# eval file. The files on disk keep all EVAL_SAMPLES, so changing this regenerates
# nothing, and a larger value later only adds the samples not yet predicted.
PRED_SAMPLES="${PRED_SAMPLES:-500}"
# QA questions reserved for evaluation. DORMANT while no qa_* task is selected: the
# --qa_start/--qa_end flags below are still passed to prepare.py, which forwards them
# only for tasks whose generator is qa.py, so they are a no-op for the current suite.
# Kept because the split is a correctness requirement, not a tuning knob, the moment a
# QA task comes back: SQuAD and HotpotQA are fixed pools (~5.9k and ~7.4k answerable
# questions), and qa.py takes questions IN ORDER from the start of the pool -- the seed
# only reshuffles distractor paragraphs, never which questions appear. Without a split,
# train and eval both start at question 0 and training cycles the whole pool, so every
# eval question and its gold answer is a training target, and a model can score by
# recalling question -> answer without reading the context. Eval takes [0, QA_HOLDOUT),
# training takes [QA_HOLDOUT, end). The margin over EVAL_SAMPLES absorbs questions
# skipped for not fitting the length budget.
QA_HOLDOUT=2000
EVAL_SEED=62                # RULER default

# ====================== TRAINING =============================================
EPOCHS=25
BATCH_SIZE=8
GRAD_ACCUM=8                # effective batch = BATCH_SIZE * GRAD_ACCUM

# Peak LR. The schedule is inverse square root (train.py): linear warmup to LR over
# WARMUP optimizer steps, then LR * sqrt(WARMUP / step).
LR=1.5e-4

# ~2 starter epochs: the starter stage (niah_single_1, niah_single_2, vt) is 3 tasks x
# TRAIN_SAMPLES x 0.9 for training = 86,400 samples = 1,350 optimizer steps per epoch at
# an effective batch of 64. train_start.py sets its own warmup as --starter_warmup_epochs
# (default 2), which tracks these sizes if they change; this value is for train.py runs.
WARMUP=2700

EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-5e-3}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-12}"

PRECISION="${PRECISION:-bf16}"
SRC_LEN=2048                # max encoder tokens during training
TGT_LEN=128                # max decoder tokens during training
# Seeds for the sweep; each arm is trained and evaluated once per seed.
# Override with e.g. SEEDS="1 2 3".
read -r -a SEEDS <<< "${SEEDS:-96 97}"
# Longest prompt + answer the decoder-only models must accept: prompts go up to the
# longest EVAL length, so the default (SRC_LEN + TGT_LEN) would truncate them at 4096 and
# 8192. Note the ALiBi model keeps a MAX_LEN x MAX_LEN causal mask per layer (~1.7 GB over
# its 6 layers at 8320, in memory and in best.pt). transformer_mask has no length limit.
MAX_LEN=$(( $(printf '%s\n' "${EVAL_SEQ_LENGTHS[@]}" | sort -n | tail -1) + TGT_LEN ))

# ====================== WANDB ================================================
WANDB_PROJECT="${WANDB_PROJECT:-Ruler_nopos}"
WANDB_ENTITY="${WANDB_ENTITY:-}"                 # empty: your default entity
WANDB_GROUP="${WANDB_GROUP:-seed_sweep}"         # one group per sweep in the UI
WANDB_MODE="${WANDB_MODE:-online}"               # online | offline | disabled

# ====================== FLAGS ================================================
DRY_RUN=false
LOCAL=false
SUMMARY=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --local)   LOCAL=true ;;
        --summary) SUMMARY=true ;;
        *) echo "ERROR: unknown argument '$arg' (expected --dry-run, --local or --summary)." >&2
           exit 1 ;;
    esac
done

# ====================== EXPERIMENT GRID ======================================
# transformer_mask arms, as explicit "<pe> <encoder mask>" pairs rather than a pe x mask
# cross product. The mask spec is one code per head, B/C/F; head order carries no
# meaning, only the count of each code. The decoder is always causal.
#
#   none       CCCCFFFF   no position encoding, 4 causal + 4 future-only heads
#   none       CCCCCCCC   no position encoding, all heads causal
#   sinusoidal BBBBBBBB   the vanilla transformer: sinusoidal PE, no mask
#
# roformer carries its own position encoding (RoPE) and no mask.
TM_ARMS=("none CCCCFFFF" "none CCCCCCCC" "sinusoidal BBBBBBBB")
OTHER_MODELS=("roformer")

for arm in "${TM_ARMS[@]}"; do
    spec="${arm#* }"
    if [ ${#spec} -ne "${TM_NUM_HEADS}" ]; then
        echo "ERROR: mask spec '${spec}' has ${#spec} codes but transformer_mask has ${TM_NUM_HEADS} heads." >&2
        exit 1
    fi
done

# Run directory / wandb run name for one cell. "-" marks a field the model does not use.
exp_name() {
    local model="$1" pe="$2" enc_mask="$3" seed="$4"
    if [ "${model}" = "transformer_mask" ]; then
        echo "transformer_mask_pe${pe}_enc${enc_mask}_s${seed}"
    else
        echo "${model}_s${seed}"
    fi
}

# Flatten into explicit "<model> <pe> <mask> <seed>" cells. Seeds are the OUTER loop, so
# the queue holds one complete replicate of every arm before the next seed starts.
EXPERIMENTS=()
for seed in "${SEEDS[@]}"; do
    for arm in "${TM_ARMS[@]}"; do
        EXPERIMENTS+=("transformer_mask ${arm} ${seed}")
    done
    for model in "${OTHER_MODELS[@]}"; do
        EXPERIMENTS+=("${model} - - ${seed}")
    done
done

# Task list (must match entries in synthetic.yaml)
source "${SCRIPT_DIR}/config_tasks.sh"
# An explicit include list, not the full synthetic[@] suite: three variable-tracking
# configurations, three multi-key needle tasks and one single-needle task. Which of them
# are trained on and which are scored are two separate lists -- see the split below. Known
# properties worth keeping in mind when reading results, since each one bears on whether
# a drop with length is positional:
#
#   vt_2chain, vt_4chain  two and four assignment chains in a noise haystack, four
#                  hops each. Both counts are fixed, so 4x the context is 4x the
#                  distance between links and nothing else -- the property that makes a
#                  2048 -> 8192 drop readable as positional. The noise haystack is one
#                  sentence repeated, so samples come out near-uniform in length and
#                  batches are barely ragged. The answer is a set of five variable
#                  names, which resists the answer-prior basin better than a single
#                  number does. vt_8chain exists in synthetic.yaml and works, but is not
#                  selected: its few-shot prefix alone eats ~31% of a 2048 context
#                  against ~25% at four chains and ~21% at two, so the overhead would
#                  vary more across the pair than the thing being measured.
#   niah_multikey_1  essay haystack with 4 key/value needles; the query must pick one.
#                  Needle count is fixed, but the essay is always the corpus prefix, so
#                  eval at 8192 contains text never seen at the 2048 training length.
#   niah_multikey_2/3  "needle" haystack: the filler is itself key/value needles, so
#                  distractor count grows with length. _3 uses uuid keys and values, so
#                  it also needs long exact copies.
#
# qa_1/qa_2 were here and were dropped. Extractive QA draws from a finite question pool
# (SQuAD ~5.9k answerable, HotpotQA ~7.4k), so 25k samples per task reuse each question
# several times, and a longer context means more distractor paragraphs rather than more
# distance -- so a drop with length is not cleanly positional. They are also the only
# tasks whose sample lengths vary by hundreds of tokens, because qa.py shrinks the
# haystack per question until it fits; that makes every batch ragged with padding.
#
# Two independent lists, the same way the sequence lengths are two independent lists.
# TRAIN_TASKS is what the model sees, EVAL_TASKS is what it is scored on, and neither is
# derived from the other. A task therefore sits in one of three places, all deliberate:
#
#   both         the normal case: an in-distribution score for a task that was trained
#   eval only    held out -- scored but never trained, a generalisation probe
#   train only   curriculum -- shapes the model, but is not a question being asked
#
# The eval-only slot gives a generalisation axis orthogonal to length:
#
#                        2048 (train len)   4096   8192
#   vt_2chain (trained)  in-distribution    len    len
#   vt_4chain (held out) distractors        both   both
#
# vt_4chain differs from vt_2chain only in the number of distractor chains -- same hop
# count, same noise haystack, same five-variable answer, same template -- so a drop from
# the second row to the first isolates distractor count the way a drop across a row
# isolates length. Doubling is the smallest step that is still a real change, which
# keeps the confound small: the few-shot prefix each sample carries grows with the chain
# count, so a wider gap would vary that overhead alongside the variable being tested. It
# costs no training time, only the eval data and one prediction pass.
#
# The train-only slot holds three CURRICULUM tasks, which shape the model but are not
# questions being asked. The previous suite was multi-key needles and multi-chain
# tracking only, and every arm floored at 0.0 on all three needle tasks -- emitting
# numbers that appear nowhere in the context (297/300, 300/300, 300/300 sampled), which
# is a copy circuit that never formed rather than one retrieving the wrong needle. The
# suite before it, which did reach 98-100, contained the easy rungs. Each of these is
# the trivial step of a ladder whose hard steps are scored, and each isolates ONE axis
# that a scored task otherwise changes two or three of at once:
#
#   niah_single_1  noise haystack, 1 needle, 7-digit answer. The cheapest possible
#                  example of "the answer is a span of the input" -- somewhere easy for
#                  the gradient to start.
#   niah_single_3  essay haystack, 1 needle, UUID answer. The only task here that
#                  teaches a ~20-token exact copy, which niah_multikey_3 requires and
#                  nothing else supplies; from niah_single_1 that is a 3-token answer
#                  becoming a 20-token one. It reached 98-100 in the suite that worked,
#                  so it is known to be learnable at this size.
#   vt             1 chain, the same job for following a chain rather than finding a
#                  span.
#
#   needles: niah_single_1 (1 needle)  ->  niah_single_3 (long copy)
#                                       -> niah_multikey_1/2/3 (pick among distractors)
#   chains:  vt (1 chain)  ->  vt_2chain (2, scored)  ->  vt_4chain (4, held out)
#
# None of the three is scored: each is a column every arm would be expected to pass, and
# the run is already 6 evaluated task-columns wide. Note the consequence -- vt is
# RULER's own shipped variable-tracking configuration, so with it unscored NO scored
# task here is unmodified RULER, and these numbers are not comparable to published ones.
# Move vt back into EVAL_TASKS if that comparison is wanted.
#
# The essay-haystack curriculum task is safe HERE in a way it would not be if scored:
# niah_single_2/3 were dropped from the scored suite because the essay haystack is
# always the corpus prefix, so evaluating at 8192 shows text never seen at 2048. That is
# an evaluation-side confound, and it simply does not arise for a task that is only ever
# trained at 2048. The train-only slot is where tasks with eval-side confounds belong.


TRAIN_TASKS=("niah_single_1" "niah_single_3" "vt" "vt_2chain"
             "niah_multikey_1" "niah_multikey_2" "niah_multikey_3")
EVAL_TASKS=("vt_2chain" "vt_4chain"
            "niah_multikey_1" "niah_multikey_2" "niah_multikey_3")

# Starter curriculum. Each cell first trains STARTER_EPOCHS epochs on STARTER_TASKS only
# (train_start.py), then continues the same model, AdamW state and LR schedule on
# TRAIN_TASKS (train_full.py) for up to EPOCHS. Every task of both lists is generated
# into the one TRAIN_DATA_DIR, and each stage loads only its own list (--starter_tasks /
# --tasks), so niah_single_2 is seen in the starter stage and never after it. LR warmup
# is STARTER_WARMUP_EPOCHS starter epochs; WARMUP applies only when STARTER_EPOCHS=0,
# which trains every cell from scratch on TRAIN_TASKS instead.
STARTER_TASKS=("niah_single_1" "niah_single_2" "vt")
STARTER_EPOCHS="${STARTER_EPOCHS:-4}"
STARTER_WARMUP_EPOCHS="${STARTER_WARMUP_EPOCHS:-2}"

# Union of the task lists, order preserving; nothing outside these loops should
# iterate several lists. TRAIN_DATA_TASKS is what gets generated for training.
_union() {
    local out=() t
    for t in "$@"; do
        case " ${out[*]:-} " in *" ${t} "*) ;; *) out+=("${t}") ;; esac
    done
    echo "${out[@]}"
}
if [ "${STARTER_EPOCHS}" -gt 0 ]; then
    read -r -a TRAIN_DATA_TASKS <<< "$(_union "${TRAIN_TASKS[@]}" "${STARTER_TASKS[@]}")"
else
    TRAIN_DATA_TASKS=("${TRAIN_TASKS[@]}")
fi
read -r -a ALL_TASKS <<< "$(_union "${TRAIN_DATA_TASKS[@]}" "${EVAL_TASKS[@]}")"

case "${PRECISION}" in
    bf16|fp16) ;;
    *) echo "ERROR: PRECISION='${PRECISION}' must be bf16 or fp16." >&2; exit 1 ;;
esac

# Fail here rather than partway through a submission: prepare.py looks each task up in
# synthetic.yaml, and a typo would otherwise surface one task into data generation.
for _t in "${ALL_TASKS[@]}"; do
    if ! grep -qE "^${_t}:" "${SCRIPT_DIR}/synthetic.yaml"; then
        echo "ERROR: task '${_t}' is not defined in ${SCRIPT_DIR}/synthetic.yaml." >&2
        exit 1
    fi
done
# Print the split rather than validate it. With two independent lists there is no
# illegal combination left to detect -- train-only and eval-only are both intended --
# but a task landing on the wrong side is silent and costs a whole run, so name each.
_both=() _train_only=() _eval_only=() _starter_only=()
for _t in "${ALL_TASKS[@]}"; do
    case " ${TRAIN_TASKS[*]} " in *" ${_t} "*) _in_train=true ;; *) _in_train=false ;; esac
    case " ${EVAL_TASKS[*]} "  in *" ${_t} "*) _in_eval=true  ;; *) _in_eval=false  ;; esac
    if   $_in_train && $_in_eval; then _both+=("${_t}")
    elif $_in_train;              then _train_only+=("${_t}")
    elif $_in_eval;               then _eval_only+=("${_t}")
    else                               _starter_only+=("${_t}")
    fi
done
echo "Tasks: ${#ALL_TASKS[@]} total"
echo "  trained and scored  : ${_both[*]:-none}"
echo "  train only (curric) : ${_train_only[*]:-none}"
echo "  eval only (held out): ${_eval_only[*]:-none}"
if [ "${STARTER_EPOCHS}" -gt 0 ]; then
    echo "  starter stage       : ${STARTER_TASKS[*]} for ${STARTER_EPOCHS} epoch(s)" \
         "(only there: ${_starter_only[*]:-none})"
else
    echo "  starter stage       : off (STARTER_EPOCHS=0)"
fi

# ====================== PATHS ================================================
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/../experiments}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
TRAIN_SCRIPT="${REPO_DIR}/train_wandb.py"

# ====================== SUMMARY MODE =========================================
if $SUMMARY; then
    # `nulls` is carried through because the metric is a plain substring match: an
    # empty prediction and a confidently wrong one both score 0.0, and only the null
    # count separates "the model emitted nothing" from "the model emitted the wrong
    # thing". Those two have completely different causes.
    echo "model,pe,encoder_mask,seed,seq_length,task,score,nulls"
    # Only this sweep's runs: their directory names end in _s<seed> (see exp_name);
    # scripts/run_experiments.sh's pe_* results in the same tree are skipped.
    for dir in "${EXP_ROOT}"/results/*_s[0-9]*/synthetic/*/pred; do
        [ -f "${dir}/summary.csv" ] || continue
        # Parse path: .../<exp_name>/synthetic/<SEQ>/pred/summary.csv
        # Shell parameter expansion only -- `grep -oP` is GNU-specific and is not
        # available in the BSD grep shipped with macOS.
        seq_dir="${dir%/pred}"          # .../synthetic/<SEQ>
        seq="${seq_dir##*/}"            # <SEQ>
        cfg_dir="${seq_dir%/*}"         # .../synthetic
        cfg_dir="${cfg_dir%/*}"         # .../<exp_name>
        cfg="${cfg_dir##*/}"            # e.g. transformer_mask_pesinusoidal_encCCCCFFFF_s42
        seed="${cfg##*_s}"
        base="${cfg%_s*}"
        case "${base}" in
            transformer_mask_pe*)
                rest="${base#transformer_mask_pe}"   # <PE>_enc<SPEC>
                model="transformer_mask"; pe="${rest%%_enc*}"; enc="${rest##*_enc}" ;;
            *)  model="${base}"; pe="-"; enc="-" ;;
        esac
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
    print(f'${model},${pe},${enc},${seed},${seq},{t},{s},{n}')
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
    "tqdm": "tqdm", "requests": "requests", "wandb": "wandb",
    "torch_geometric": "torch_geometric",
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
        echo "           PIP_ARGS='--index-url https://<internal-mirror>/simple' bash run_seed_experiments.sh" >&2
        echo "           PIP_ARGS='--no-index --find-links \$HOME/wheels' bash run_seed_experiments.sh" >&2
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
                echo "           TORCH_SPEC='torch --index-url https://download.pytorch.org/whl/cu121' bash run_seed_experiments.sh" >&2
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
        f"       GPU type the wheel supports (see GPU_SPEC in run_seed_experiments.sh), or\n"
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

    # Only fetch what the selected tasks actually read. Most needle tasks use the noise
    # or needle haystacks and need no corpus at all, so a run restricted to those
    # should not pull down three datasets it will never open.
    # Which tasks use the essay haystack is set in synthetic.yaml (type_haystack: essay).
    # Must track every task with `type_haystack: essay` in synthetic.yaml. niah_single_3
    # was missing here, which went unnoticed only because niah_single_2 was always
    # selected alongside it; on a task list with single_3 but not single_2 the corpus
    # was never fetched and generation failed on the missing file.
    local essay_tasks=" niah_single_2 niah_single_3 niah_multikey_1 niah_multivalue niah_multiquery "
    local want_essay=false want_qa=false
    for t in "${ALL_TASKS[@]}"; do
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
        for TASK in "${EVAL_TASKS[@]}"; do
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
        # TRAIN_DATA_TASKS (train + starter), never an eval-only task. Each stage also
        # filters to its own list, but an eval task generated here with the TRAIN seed
        # would be one flag away from leaking into training.
        for TASK in "${TRAIN_DATA_TASKS[@]}"; do
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
# The training stages each cell runs, in order: the starter curriculum (starter, then
# full) or, with STARTER_EPOCHS=0, a single from-scratch run.
train_stages() {
    if [ "${STARTER_EPOCHS}" -gt 0 ]; then echo "starter full"; else echo "scratch"; fi
}

# Fills TRAIN_CMD with the command for one stage of one cell. Shared by run_experiment
# and --dry-run, so what the dry run prints is exactly what a job runs.
train_command() {
    local model="$1" pe="$2" enc_mask="$3" seed="$4" stage="$5"
    local EXP_NAME; EXP_NAME="$(exp_name "${model}" "${pe}" "${enc_mask}" "${seed}")"
    TRAIN_CMD=(python "${TRAIN_SCRIPT}"
        --stage        "${stage}"
        --model        "${model}")
    local tags=("model:${model}" "seed:${seed}" "stage:${stage}")
    local run_name="${EXP_NAME}"
    if [ "${stage}" = "starter" ]; then run_name+="-starter"; fi
    case "${stage}" in
        starter)
            TRAIN_CMD+=(--starter_tasks "${STARTER_TASKS[@]}"
                        --starter_epochs "${STARTER_EPOCHS}"
                        --starter_warmup_epochs "${STARTER_WARMUP_EPOCHS}")
            ;;
        full)
            TRAIN_CMD+=(--init_from "${EXP_ROOT}/${EXP_NAME}/starter.pt"
                        --tasks "${TRAIN_TASKS[@]}")
            ;;
        scratch)
            TRAIN_CMD+=(--tasks "${TRAIN_TASKS[@]}")
            ;;
    esac
    if [ "${model}" = "transformer_mask" ]; then
        TRAIN_CMD+=(--pe "${pe}" --encoder_mask "${enc_mask}")
        tags+=("pe:${pe}" "mask:${enc_mask}")
    fi
    TRAIN_CMD+=(
        --max_len      "${MAX_LEN}"
        --dropout      "${DROPOUT}"
        --data_format  ruler
        --data_dir     "${TRAIN_DATA_DIR}"
        --tokenizer    "${TOKENIZER}"
        --src_len      "${SRC_LEN}"
        --tgt_len      "${TGT_LEN}"
        --epochs       "${EPOCHS}"
        --batch_size   "${BATCH_SIZE}"
        --grad_accum   "${GRAD_ACCUM}"
        --early_stop_patience   "${EARLY_STOP_PATIENCE}"
        --early_stop_min_delta  "${EARLY_STOP_MIN_DELTA}"
        --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}"
        --lr           "${LR}"
        --warmup_steps "${WARMUP}"
        --seed         "${seed}"
        "--${PRECISION}"
        --output_dir   "${EXP_ROOT}/${EXP_NAME}"
        --project      "${WANDB_PROJECT}"
        --group        "${WANDB_GROUP}"
        --run_name     "${run_name}"
        --tags         "${tags[@]}"
        --mode         "${WANDB_MODE}")
    if [ -n "${WANDB_ENTITY}" ]; then
        TRAIN_CMD+=(--entity "${WANDB_ENTITY}")
    fi
}

run_experiment() {
    local model="$1" pe="$2" enc_mask="$3" seed="$4"
    local EXP_NAME; EXP_NAME="$(exp_name "${model}" "${pe}" "${enc_mask}" "${seed}")"
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
        local stage
        for stage in $(train_stages); do
            # starter.pt is written once, after the last starter epoch, so finding it
            # means the starter stage finished; a resubmitted job goes straight on.
            if [ "${stage}" = "starter" ] && ! $RETRAIN && [ -f "${EXP_DIR}/starter.pt" ]; then
                echo "--- Reusing ${EXP_DIR}/starter.pt (starter stage already completed) ---"
                continue
            fi
            echo "--- Training: ${stage} stage ---"
            train_command "${model}" "${pe}" "${enc_mask}" "${seed}" "${stage}"
            "${TRAIN_CMD[@]}"
        done
    fi

    # ---- Phase 2: Evaluate at each RULER sequence length ------------------
    for SEQ_LEN in "${EVAL_SEQ_LENGTHS[@]}"; do
        local RESULTS="${EXP_ROOT}/results/${EXP_NAME}/synthetic/${SEQ_LEN}"
        # Read the shared eval data rather than regenerating it per configuration.
        local EVAL_DATA="${EVAL_DATA_ROOT}/${SEQ_LEN}/data"
        local PRED_DIR="${RESULTS}/pred"
        mkdir -p "${PRED_DIR}"

        for TASK in "${EVAL_TASKS[@]}"; do
            # predict
            local CAP
            CAP="$(answer_token_cap "${EVAL_DATA}/${TASK}/validation.jsonl")"
            python "${SCRIPT_DIR}/pred/call_api.py" \
                --max_new_tokens "${CAP}" \
                --num_samples "${PRED_SAMPLES}" \
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

    # Every eval length is done and scored, so the weights have served their purpose:
    # predictions and summaries are on disk and are what the analysis reads. Deleting
    # here is safe because 'set -e' aborts before this line if any eval failed, so a
    # checkpoint is only removed once its results exist.
    if ! $KEEP_CHECKPOINTS; then
        # starter.pt too: it carries the AdamW state, so it is ~3x the size of best.pt.
        rm -f "${CKPT}" "${EXP_DIR}/last.pt" "${EXP_DIR}/starter.pt"
        echo "    removed checkpoint (KEEP_CHECKPOINTS=false); results are in ${EXP_ROOT}/results/${EXP_NAME}"
    fi

    echo "=== Done: ${EXP_NAME} ==="
}

# ====================== WANDB PREFLIGHT ======================================
# Fail before submitting rather than in every job: an online run with no credentials
# makes each of the jobs die at wandb.init after queueing.
if [ "${WANDB_MODE}" = "online" ] && ! $DRY_RUN && [ -z "${WANDB_API_KEY:-}" ] \
        && ! grep -qs "api.wandb.ai" "${HOME}/.netrc"; then
    echo "ERROR: WANDB_MODE=online but no wandb credentials were found (no WANDB_API_KEY," >&2
    echo "       no api.wandb.ai entry in ~/.netrc). Run 'wandb login' on a home the" >&2
    echo "       compute nodes share, export WANDB_API_KEY, or set WANDB_MODE=offline." >&2
    exit 1
fi

# ====================== SHARED DATA PREP JOB =================================
# Training and eval data are identical across all cells and seeds. Generating them
# inside every job would have many processes writing the same files concurrently, and
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
$(declare -p AUTO_INSTALL VENV_DIR REQUIREMENTS BOOTSTRAP_PYTHON PIP_ARGS TORCH_SPEC SCRIPT_DIR CORPUS_DIR TOKENIZER ALL_TASKS TRAIN_TASKS EVAL_TASKS \
             TRAIN_DATA_TASKS TRAIN_DATA_DIR TRAIN_SEQ_LENGTHS TRAIN_SAMPLES TRAIN_SEED \
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
    read -r model pe enc_mask seed <<< "${cell}"
    EXP_NAME="$(exp_name "${model}" "${pe}" "${enc_mask}" "${seed}")"

    if $LOCAL; then
        # ---------- local: prepare shared data once, then run each experiment ---
        if [ $n_jobs -eq 0 ]; then
            setup_env
            check_environment
            fetch_corpora
            generate_train_data
            generate_eval_data
        fi
        run_experiment "${model}" "${pe}" "${enc_mask}" "${seed}"
        n_jobs=$((n_jobs + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[dry-run] ${EXP_NAME}"
        for stage in $(train_stages); do
            train_command "${model}" "${pe}" "${enc_mask}" "${seed}" "${stage}"
            printf '    %q' "${TRAIN_CMD[@]}"; echo
        done
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
$(declare -p KEEP_CHECKPOINTS RETRAIN PRED_SAMPLES)
$(declare -p EXP_ROOT TRAIN_DATA_DIR EVAL_DATA_ROOT TRAIN_SCRIPT MAX_LEN TOKENIZER \
             SRC_LEN TGT_LEN EPOCHS BATCH_SIZE GRAD_ACCUM LR WARMUP PRECISION \
             EVAL_SEQ_LENGTHS EVAL_SAMPLES EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA \
             EARLY_STOP_MIN_EPOCHS DROPOUT EVAL_SEED QA_HOLDOUT EVAL_TASKS \
             TRAIN_TASKS STARTER_TASKS STARTER_EPOCHS STARTER_WARMUP_EPOCHS \
             REPO_DIR SCRIPT_DIR WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_MODE)
$(declare -f setup_env)
$(declare -f _missing_packages)
$(declare -f _pip_install)
$(declare -f check_environment)
$(declare -f check_gpu)
$(declare -f answer_token_cap)
$(declare -f exp_name)
$(declare -f train_stages)
$(declare -f train_command)
$(declare -f run_experiment)

setup_env

echo "Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# Data is produced by the prep job this one depends on; nothing to generate here.
check_environment
check_gpu
run_experiment "${model}" "${pe}" "${enc_mask}" "${seed}"
SLURM_EOF

    echo "  -> submitted ${EXP_NAME}"
    n_jobs=$((n_jobs + 1))
done

echo ""
echo "========================================================"
echo "  ${n_jobs} runs dispatched (${#SEEDS[@]} seeds: ${SEEDS[*]})"
echo "  wandb     : project ${WANDB_PROJECT}, group ${WANDB_GROUP} (${WANDB_MODE})"
echo "  Results   : ${EXP_ROOT}/results/"
echo "  Summaries : bash $0 --summary"
echo ""
echo "  Monitor   : squeue -u \$USER"
echo "  Logs      : ${LOG_DIR}/"
echo "========================================================"
