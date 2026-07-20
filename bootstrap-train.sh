#!/bin/bash
# Bootstrap script for Agent Execution Partnership autoresearch training pipeline.
#
# Automates the full workflow:
#   1. Install training dependencies (torch, rustbpe, tiktoken, etc.)
#   2. Download dataset shards and train tokenizer
#   3. Establish baseline BPB
#   4. Run autonomous experiments
#
# Usage:
#   ./bootstrap-train.sh                    # defaults: depth=8, iterations=20
#   ./bootstrap-train.sh --depth 12 --iterations 50 --shards 200
#   ./bootstrap-train.sh --skip-baseline
#
# Requirements: Python 3.12+, CUDA 12.1+, ~100GB storage

set -e

# Defaults
DEPTH=8
ITERATIONS=20
SHARDS=100
SKIP_BASELINE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --shards)
            SHARDS="$2"
            shift 2
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
readonly CYAN='\033[0;36m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m' # No Color

header() {
    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..80})${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..80})${NC}"
    echo ""
}

step() {
    local step=$1
    local total=$2
    local text=$3
    echo -e "${GREEN}[${step}/${total}] ${text}${NC}"
}

run_step() {
    local name=$1
    local cmd=$2
    local step=$3
    local total=$4
    
    step "$step" "$total" "Running: $name"
    if ! eval "$cmd"; then
        echo -e "${RED}FAILED: $name${NC}"
        exit 1
    fi
}

STEP=1
TOTAL_STEPS=5

header "AEP Autoresearch Training Bootstrap"

echo "Configuration:"
echo "  Depth:        $DEPTH layers"
echo "  Iterations:   $ITERATIONS experiments"
echo "  Shards:       $SHARDS training shards"
echo "  Baseline:     $([ "$SKIP_BASELINE" = true ] && echo 'skipped' || echo 'run first')"
echo ""

# Step 1: Install training deps
run_step "Install training dependencies" \
    "python -m pip install -e '.[train]' --quiet" \
    "$STEP" "$TOTAL_STEPS"
((STEP++))

# Step 2: Prepare data
run_step "Download dataset and train tokenizer" \
    "aep research prepare --shards $SHARDS" \
    "$STEP" "$TOTAL_STEPS"
((STEP++))

# Step 3: Establish baseline (unless skipped)
if [ "$SKIP_BASELINE" = false ]; then
    run_step "Establish baseline BPB" \
        "aep research train --baseline --depth $DEPTH" \
        "$STEP" "$TOTAL_STEPS"
else
    step "$STEP" "$TOTAL_STEPS" "SKIPPED: Baseline (use previous baseline from ledger)"
fi
((STEP++))

# Step 4: Run autonomous experiments
run_step "Run $ITERATIONS autonomous experiments" \
    "aep research train --depth $DEPTH --iterations $ITERATIONS" \
    "$STEP" "$TOTAL_STEPS"

header "✓ Training pipeline complete"

echo -e "${YELLOW}Results:${NC}"
echo "  Experiment ledger: research/aee-autoresearch/experiment_ledger.jsonl"
echo "  Results TSV:       research/aee-autoresearch/results.tsv"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review experiment_ledger.jsonl to analyze results"
echo "  2. Check results.tsv for training metrics (BPB, memory, MFU)"
echo "  3. Retained experiments are new baselines for next run"
echo ""
