# Agent Execution Partnership — Examples

This directory contains examples and demonstrations of the AEP control plane in action.

## Demo Contents

- **index.html** — Interactive browser demonstration of the closed-loop execution cycle
  - Real-time action proposal, policy evaluation, execution, and verification
  - Safe local-only operations (no real backend connectivity)

- **action_request.json** — Example action proposal structure
- **task_contract.json** — Example task contract
- **audit-ledger.jsonl** — Sample audit trail from a completed execution

## Running the Local Demo Site

### 1. Start the demo server

From the root `agent-execution-partnership/` directory:

```powershell
# Install dependencies (one time)
py -m pip install -e .[dev]

# Run the API server
aep serve
```

The server starts on `http://localhost:8000`.

### 2. Open the demo dashboard

```powershell
# In a browser, navigate to:
# http://localhost:8000/demo
```

Or open `index.html` directly in a browser:

```powershell
# Windows
start examples/local-demo-site/index.html

# macOS
open examples/local-demo-site/index.html

# Linux
xdg-open examples/local-demo-site/index.html
```

## Training Pipeline Example

The local demo also showcases the autonomous research loop that trains a GPT model using the control plane's execution guarantees.

### Quick Start (Full Training Pipeline)

From the root directory, run the bootstrap script:

**Windows:**
```powershell
.\bootstrap-train.ps1
```

**Linux/macOS:**
```bash
chmod +x bootstrap-train.sh
./bootstrap-train.sh
```

This orchestrates:
1. **Installation** — Training dependencies (torch, rustbpe, tiktoken, etc.)
2. **Data Preparation** — Downloads 100 training shards and trains a BPE tokenizer
3. **Baseline** — Establishes baseline bits-per-byte (BPB) metric
4. **Experiments** — Runs 20 autonomous experiments with automatic policy-driven evaluation

### Makefile Targets

```powershell
# All-in-one
make train-all

# Or step-by-step
make train-install      # Install training deps
make train-prepare      # Download data + train tokenizer
make train-baseline     # Establish baseline BPB
make train-run          # Run 20 experiments
```

### Direct CLI

```powershell
# Prepare data
aep research prepare --shards 100

# Run baseline with depth=8 model
aep research train --baseline --depth 8

# Run 20 experiments
aep research train --depth 8 --iterations 20

# Custom config (depth=12, 50 iterations)
aep research train --depth 12 --iterations 50
```

### Results Analysis

After training completes, results are available in two formats:

**JSON Ledger** (`research/aee-autoresearch/experiment_ledger.jsonl`):
```json
{
  "experiment_id": "exp-abc-123",
  "commit": "3a7f9d2",
  "val_bpb": 1.234,
  "training_seconds": 298.5,
  "peak_vram_mb": 8234.0,
  "decision": "retain",
  "reason": "bpb_improved_by_0.020000"
}
```

**TSV Metrics** (`research/aee-autoresearch/results.tsv`):
```
commit      val_bpb  memory_gb  mfu_percent  status   description
3a7f9d2     1.254    8.2        42.3         retain   baseline
5b8e1c4     1.234    8.1        42.5         retain   +0.02 improvement
7c9f2d5     1.235    8.3        41.8         revert   -0.001 regression
```

### Model Configuration

- **Depth:** 8 layers (configurable)
- **Budget:** 300 seconds per run (fixed)
- **Parameters:** ~50M (varies by depth)
- **Metric:** Bits-per-byte (BPB) — lower is better

Experiments that improve BPB by ≥0.001 bits are automatically retained as the new baseline.

### Understanding the Control Plane Integration

The training loop demonstrates AEP's core guarantees:

1. **Authorization** — Policy engine evaluates model checkpoints before retention
2. **Observability** — All training metrics logged to immutable audit trail
3. **Verification** — Bit-per-byte metric compared to expected improvement threshold
4. **Recovery** — Regressing experiments automatically reverted (no manual intervention)

## Example Task Flows

### Reading the Examples

```powershell
# View action structure
cat examples/action_request.json

# View task contract
cat examples/task_contract.json

# View audit trail
cat examples/audit-ledger.jsonl
```

### Schemas

See `schemas/` directory for JSON Schema definitions:

- `schemas/action_request.schema.json` — Action proposal structure
- `schemas/task_contract.schema.json` — Task contract structure
- `schemas/execution_evidence.schema.json` — Execution result format
- `schemas/verification_result.schema.json` — Verification outcome format

## Next Steps

1. **Run the local demo** — Interactive browser UI shows the full execution flow
2. **Start training** — `.\bootstrap-train.ps1` for end-to-end pipeline
3. **Analyze results** — Review `experiment_ledger.jsonl` and `results.tsv`
4. **Customize** — Adjust depth, iterations, or data shards for your needs

## Troubleshooting

**Missing dependencies?**
```powershell
py -m pip install -e ".[train]"
```

**CUDA not available?**
Training requires GPU. Without CUDA, you'll see helpful error with setup instructions.

**Port 8000 in use?**
```powershell
# The API server will automatically find another port
aep serve --port 8001
```

**Results not appearing?**
Training outputs are in `research/aee-autoresearch/`:
```powershell
# Check files exist
ls research/aee-autoresearch/
```

## References

- [README.md](../../README.md) — Main project documentation
- [docs/evaluation.md](../../docs/evaluation.md) — Evaluation metrics and methodology
- [docs/deployment.md](../../docs/deployment.md) — Production deployment guide
- [autoresearch-master/program.md](../../autoresearch-master/program.md) — Training algorithm specification
