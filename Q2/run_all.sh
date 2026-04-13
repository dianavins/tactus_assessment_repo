#!/usr/bin/env bash
# run_all.sh — end-to-end DTD texture classifier pipeline
#
# Run from the Q2/ directory:
#   bash run_all.sh
#
# Optional flags:
#   --skip_install        skip pip install
#   --skip_expand         skip dataset expansion (if already done)
#   --skip_dinov2         skip DINOv2 probe (requires internet or --dinov2_local)
#   --skip_distill        skip knowledge distillation
#   --skip_eval           skip Step 5 evaluation (use existing results.json)
#   --skip_densenet       skip DenseNet-201 training
#   --small_only          train MobileNetV3 only (skip ConvNeXt)
#   --large_only          train ConvNeXt only (skip MobileNetV3 + distillation)
#   --freq_mix            enable frequency-domain mixing in dataset expansion
#   --dinov2_local PATH   path to local dinov2_vits14.pth
#   --batch_size N        override batch size (default 32)

set -euo pipefail

# Force UTF-8 output so non-ASCII characters (arrows, progress bars, etc.)
# don't crash on Windows consoles using cp1252.
export PYTHONIOENCODING=utf-8

# ── Locate Python ──────────────────────────────────────────────────────────
# Find the first Python executable that has pip. On Windows/Git Bash the system
# Python (e.g. /usr/bin/python3) often lacks pip; the real install is elsewhere.
# Honour a caller-supplied $PYTHON override if set.
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=""
    for candidate in python python3 py /c/Python312/python /c/Python311/python /c/Python310/python; do
        if command -v "$candidate" &>/dev/null; then
            if "$candidate" -m pip --version &>/dev/null 2>&1; then
                PYTHON="$candidate"
                break
            fi
        fi
    done
fi
if [[ -z "$PYTHON" ]]; then
    die "No Python with pip found. Set PYTHON=/path/to/python before running, or add it to PATH."
fi
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ── Colour helpers ─────────────────────────────────────────────────────────
BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RED="\033[31m"

step()  { echo -e "\n${BOLD}${CYAN}▶ $*${RESET}"; }
ok()    { echo -e "${GREEN}✓ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }
die()   { echo -e "${RED}✗ $*${RESET}"; exit 1; }

elapsed() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "%dm%02ds" $(( secs/60 )) $(( secs%60 ))
}

# ── Parse flags ────────────────────────────────────────────────────────────
SKIP_INSTALL=0; SKIP_EXPAND=0; SKIP_DINOV2=0; SKIP_DISTILL=0; SKIP_DENSENET=0; SKIP_EVAL=0
SMALL_ONLY=0;   LARGE_ONLY=0;  FREQ_MIX_FLAG=""
DINOV2_LOCAL=""; BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_install)  SKIP_INSTALL=1  ;;
        --skip_expand)   SKIP_EXPAND=1   ;;
        --skip_dinov2)   SKIP_DINOV2=1   ;;
        --skip_distill)   SKIP_DISTILL=1   ;;
        --skip_eval)      SKIP_EVAL=1      ;;
        --skip_densenet)  SKIP_DENSENET=1  ;;
        --small_only)     SMALL_ONLY=1     ;;
        --large_only)    LARGE_ONLY=1; SKIP_DISTILL=1 ;;
        --freq_mix)      FREQ_MIX_FLAG="--freq_mix" ;;
        --dinov2_local)  DINOV2_LOCAL="$2"; shift ;;
        --batch_size)    BATCH_SIZE="$2"; shift ;;
        *) die "Unknown flag: $1" ;;
    esac
    shift
done

# ── Paths (all relative to Q2/) ────────────────────────────────────────────
DATA_ROOT="./data"
EXPANDED_ROOT="./data/dtd_expanded"
CKPT="./checkpoints"
LOGS="./runs"
RESULTS="results.json"

mkdir -p "$CKPT" "$LOGS"

START_TIME=$(date +%s)
LOGFILE="run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "Pipeline started at $(date). Log: $LOGFILE"

# ── Step 0: Install dependencies ───────────────────────────────────────────
if [[ $SKIP_INSTALL -eq 0 ]]; then
    step "Step 0: Installing dependencies"
    "$PYTHON" -m pip install -r requirements.txt
    ok "Dependencies installed ($(elapsed))"
else
    warn "Step 0: Skipping pip install"
fi

# ── Step 1: Expand dataset ─────────────────────────────────────────────────
if [[ $SKIP_EXPAND -eq 0 ]]; then
    step "Step 1: Expanding DTD training set (multi-crop + optional freq mix)"
    "$PYTHON" expand_dataset.py \
        --data_root "$DATA_ROOT" \
        --expanded_root "$EXPANDED_ROOT" \
        --split_id 1 \
        $FREQ_MIX_FLAG
    ok "Dataset expanded ($(elapsed))"
else
    warn "Step 1: Skipping dataset expansion"
fi

# ── Helper: run a training step only if its final checkpoint is absent ─────
# Usage: train_if_needed <final_ckpt> <step_label> <command...>
train_if_needed() {
    local final_ckpt="$1"; shift
    local label="$1";      shift
    if [[ -f "$final_ckpt" ]]; then
        warn "$label already complete — found $(basename "$final_ckpt"), skipping."
    else
        step "$label"
        "$@"
        ok "$label done ($(elapsed))"
    fi
}

# ── Step 2: Train ConvNeXt-Tiny ────────────────────────────────────────────
if [[ $SMALL_ONLY -eq 0 ]]; then
    train_if_needed "$CKPT/convnext_stage1_ema_final.pth" \
        "Step 2a: ConvNeXt — Stage 1 (head only, frozen backbone)" \
        "$PYTHON" train.py \
            --model convnext --stage 1 \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/convnext_stage2_ema_final.pth" \
        "Step 2b: ConvNeXt — Stage 2 (last 2 blocks, layer-wise LR)" \
        "$PYTHON" train.py \
            --model convnext --stage 2 \
            --resume "$CKPT/convnext_stage1_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/convnext_stage3_ema_final.pth" \
        "Step 2c: ConvNeXt — Stage 3 (full network, low LR)" \
        "$PYTHON" train.py \
            --model convnext --stage 3 \
            --resume "$CKPT/convnext_stage2_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"
fi

# ── Step 3: Train MobileNetV3-Large ───────────────────────────────────────
if [[ $LARGE_ONLY -eq 0 ]]; then
    train_if_needed "$CKPT/mobilenetv3_stage1_ema_final.pth" \
        "Step 3a: MobileNetV3 — Stage 1 (head only)" \
        "$PYTHON" train.py \
            --model mobilenetv3 --stage 1 \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/mobilenetv3_stage2_ema_final.pth" \
        "Step 3b: MobileNetV3 — Stage 2 (last feature groups)" \
        "$PYTHON" train.py \
            --model mobilenetv3 --stage 2 \
            --resume "$CKPT/mobilenetv3_stage1_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/mobilenetv3_stage3_ema_final.pth" \
        "Step 3c: MobileNetV3 — Stage 3 (full network)" \
        "$PYTHON" train.py \
            --model mobilenetv3 --stage 3 \
            --resume "$CKPT/mobilenetv3_stage2_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"
fi

# ── Step 3b: Train DenseNet-201 ───────────────────────────────────────────
if [[ $SKIP_DENSENET -eq 0 && $LARGE_ONLY -eq 0 ]]; then
    train_if_needed "$CKPT/densenet201_stage1_ema_final.pth" \
        "Step 3b-1: DenseNet-201 — Stage 1 (head only, frozen backbone)" \
        "$PYTHON" train.py \
            --model densenet201 --stage 1 \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/densenet201_stage2_ema_final.pth" \
        "Step 3b-2: DenseNet-201 — Stage 2 (last 2 block groups)" \
        "$PYTHON" train.py \
            --model densenet201 --stage 2 \
            --resume "$CKPT/densenet201_stage1_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/densenet201_stage3_ema_final.pth" \
        "Step 3b-3: DenseNet-201 — Stage 3 (full network)" \
        "$PYTHON" train.py \
            --model densenet201 --stage 3 \
            --resume "$CKPT/densenet201_stage2_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"
else
    warn "Step 3b: Skipping DenseNet-201 training"
fi

# ── Step 4: DINOv2 linear probe ────────────────────────────────────────────
if [[ $SKIP_DINOV2 -eq 0 ]]; then
    DINOV2_ARGS=""
    [[ -n "$DINOV2_LOCAL" ]] && DINOV2_ARGS="--dinov2_local $DINOV2_LOCAL"
    train_if_needed "$CKPT/dinov2_probe_stage1_ema_final.pth" \
        "Step 4: DINOv2 ViT-S/14 linear probe (frozen backbone)" \
        "$PYTHON" train.py \
            --model dinov2_probe --stage 1 \
            --no_expanded \
            --data_root "$DATA_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE" \
            $DINOV2_ARGS
else
    warn "Step 4: Skipping DINOv2 probe"
fi

# ── Step 5: Evaluate all trained models ────────────────────────────────────
if [[ $SKIP_EVAL -eq 1 ]]; then
    warn "Step 5: Skipping evaluation (using existing $RESULTS)"
else
step "Step 5: Evaluating all models on test split"

if [[ $SMALL_ONLY -eq 0 ]]; then
    "$PYTHON" evaluate.py \
        --checkpoint "$CKPT/convnext_stage3_ema_final.pth" \
        --model convnext --ema --latency \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS"
    ok "ConvNeXt evaluated"
fi

if [[ $LARGE_ONLY -eq 0 ]]; then
    "$PYTHON" evaluate.py \
        --checkpoint "$CKPT/mobilenetv3_stage3_ema_final.pth" \
        --model mobilenetv3 --ema --latency \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS"
    ok "MobileNetV3 evaluated"
fi

if [[ $SKIP_DINOV2 -eq 0 ]]; then
    DINOV2_ARGS=""
    [[ -n "$DINOV2_LOCAL" ]] && DINOV2_ARGS="--dinov2_local $DINOV2_LOCAL"
    "$PYTHON" evaluate.py \
        --checkpoint "$CKPT/dinov2_probe_stage1_ema_final.pth" \
        --model dinov2_probe --ema \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS" \
        $DINOV2_ARGS
    ok "DINOv2 probe evaluated"
fi

if [[ $SKIP_DENSENET -eq 0 && $LARGE_ONLY -eq 0 ]]; then
    "$PYTHON" evaluate.py \
        --checkpoint "$CKPT/densenet201_stage3_ema_final.pth" \
        --model densenet201 --ema --latency \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS"
    ok "DenseNet-201 evaluated"
fi

ok "All evaluations done ($(elapsed))"
fi  # end SKIP_EVAL check

# ── Step 6: Knowledge distillation (best teacher → MobileNetV3) ───────────
if [[ $SKIP_DISTILL -eq 0 && $SMALL_ONLY -eq 0 && $LARGE_ONLY -eq 0 ]]; then

    # Select highest top-1 model as teacher (excluding mobilenetv3, the student)
    TEACHER_MODEL=$("$PYTHON" -c "
import json, sys
with open('$RESULTS') as f: d = json.load(f)
candidates = {k: v.get('top1_fp32', 0) for k, v in d.items() if k != 'mobilenetv3'}
if not candidates: sys.exit('No teacher candidates found in $RESULTS')
best = max(candidates, key=candidates.get)
print(best)
")
    case "$TEACHER_MODEL" in
        convnext)     TEACHER_CKPT="$CKPT/convnext_stage3_ema_final.pth" ;;
        dinov2_probe) TEACHER_CKPT="$CKPT/dinov2_probe_stage1_ema_final.pth" ;;
        densenet201)  TEACHER_CKPT="$CKPT/densenet201_stage3_ema_final.pth" ;;
        *) die "Unknown teacher model: $TEACHER_MODEL" ;;
    esac
    step "Selected teacher: $TEACHER_MODEL ($(get_top1 "$TEACHER_MODEL") top-1) → $TEACHER_CKPT"

    train_if_needed "$CKPT/mobilenetv3_distil_stage1_ema_final.pth" \
        "Step 6a: Distillation — Stage 1 (student head only)" \
        "$PYTHON" distill.py \
            --teacher "$TEACHER_CKPT" \
            --teacher_model "$TEACHER_MODEL" \
            --stage 1 \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/mobilenetv3_distil_stage2_ema_final.pth" \
        "Step 6b: Distillation — Stage 2" \
        "$PYTHON" distill.py \
            --teacher "$TEACHER_CKPT" \
            --teacher_model "$TEACHER_MODEL" \
            --stage 2 \
            --resume "$CKPT/mobilenetv3_distil_stage1_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    train_if_needed "$CKPT/mobilenetv3_distil_stage3_ema_final.pth" \
        "Step 6c: Distillation — Stage 3" \
        "$PYTHON" distill.py \
            --teacher "$TEACHER_CKPT" \
            --teacher_model "$TEACHER_MODEL" \
            --stage 3 \
            --resume "$CKPT/mobilenetv3_distil_stage2_best.pth" \
            --data_root "$DATA_ROOT" --expanded_root "$EXPANDED_ROOT" \
            --checkpoint_dir "$CKPT" --log_dir "$LOGS" \
            --batch_size "$BATCH_SIZE"

    step "Step 6d: Evaluating distilled MobileNetV3"
    "$PYTHON" evaluate.py \
        --checkpoint "$CKPT/mobilenetv3_distil_stage3_ema_final.pth" \
        --model mobilenetv3 --ema --latency \
        --data_root "$DATA_ROOT" \
        --output "$RESULTS"
    ok "Distilled model evaluated ($(elapsed))"
else
    warn "Step 6: Skipping knowledge distillation"
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Pipeline complete in $(elapsed).${RESET}"
echo "Results: $RESULTS"
echo "Log:     $LOGFILE"
echo ""
