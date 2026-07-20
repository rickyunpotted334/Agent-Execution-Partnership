# Local Demo Site

Interactive browser-based demonstration of Agent Execution Partnership's closed-loop execution model.

## What This Demo Shows

The demo visualizes the complete AEP lifecycle in real-time:

1. **Goal Definition** — Agent specifies desired outcome
2. **Task Contract** — Breakdown into bounded actions
3. **State Observation** — Pre-execution environment snapshot
4. **Action Channel Selection** — Route to appropriate executor
5. **Policy Evaluation** — Authorization and risk classification
6. **Capability Checks** — Verify prerequisites and boundaries
7. **Execution** — Run one bounded action
8. **Post-Execution Observation** — Capture new state
9. **Verification** — Validate expected effects occurred
10. **Anomaly Detection** — Flag unexpected side-effects
11. **Recovery/Escalation** — Handle failures automatically

## Running the Demo

### Option 1: Via API Server (Recommended)

```powershell
# From root directory, start the server
aep serve

# Open in browser
start http://localhost:8000/demo
```

The API server provides real execution context and audit logging.

### Option 2: Static File (Offline)

```powershell
# Open directly in browser
start index.html
```

Or use a local HTTP server to avoid CORS issues:

```powershell
# Python 3.12+
py -m http.server 8080 --directory examples/local-demo-site

# Then navigate to: http://localhost:8080
```

## Demo Controls

### Simulate Actions

The demo provides pre-loaded examples:

- **Read Action** — Safe, read-only (auto-approved)
- **Reversible Write** — Audit + verification required
- **Consequential Write** — Manual approval required
- **Financial Transaction** — Explicit human sign-off

Click "Propose Action" to step through the execution flow.

### View Audit Trail

The "Audit" tab shows all executed actions with:
- Timestamp and action ID
- Policy decision and reason
- Approval chain
- Execution evidence
- Verification status

### Check State Changes

The "State" tab displays:
- Pre-execution baseline
- Post-execution snapshot
- Delta (what changed)
- Expected vs. actual effects

## Training Pipeline Integration

The demo also showcases autonomous training experiments using AEP's execution control:

```powershell
# From root directory
.\bootstrap-train.ps1
```

This runs:
1. Data preparation (download shards + train tokenizer)
2. Baseline model training (5 minutes)
3. Autonomous experiment loop (20 iterations × 5 minutes)
4. Policy-driven evaluation (retain/revert decisions)

**Results** appear in:
- `research/aee-autoresearch/experiment_ledger.jsonl` — experiment records
- `research/aee-autoresearch/results.tsv` — metric table

The training loop demonstrates how AEP enforces:
- **Authorization** — Only approved model updates survive
- **Observability** — All training metrics audited
- **Verification** — Metrics compared to baseline
- **Recovery** — Degrading models automatically reverted

## File Structure

```
local-demo-site/
├── index.html           # Main demo page (open in browser)
├── README.md           # This file
```

Connected resources (in parent `examples/` directory):
```
examples/
├── README.md                    # Full examples guide
├── action_request.json          # Example action proposal
├── task_contract.json           # Example task contract
├── audit-ledger.jsonl          # Sample audit trail
└── local-demo-site/
    └── index.html
```

## Browser Compatibility

✓ Chrome/Edge 90+
✓ Firefox 88+
✓ Safari 14+
✓ Mobile browsers (responsive design)

## Keyboard Shortcuts

- `Space` — Play/pause animation
- `←` / `→` — Previous/next step
- `H` — Toggle help overlay
- `A` — Jump to audit view
- `R` — Reset to initial state

## Troubleshooting

### Demo won't load

1. Check API server is running: `aep serve`
2. Try direct file: `start index.html`
3. Clear browser cache: `Ctrl+Shift+Del` → Clear All

### Can't connect to API

1. Verify server is on `http://localhost:8000`
2. Check firewall isn't blocking port 8000
3. Try alternate port: `aep serve --port 8001`

### Training results not showing

1. Confirm `aep research prepare` completed
2. Check `research/aee-autoresearch/` directory exists
3. Verify torch is installed: `py -c "import torch; print(torch.__version__)"`

## Next Steps

1. **Run the demo** — Click through the visualization
2. **Start training** — `.\bootstrap-train.ps1`
3. **Review results** — Open `experiment_ledger.jsonl` with a text editor or pandas
4. **Customize** — Modify experiment parameters (depth, iterations, shards)

## References

- [API Docs](../../docs/api.md)
- [Architecture Guide](../../docs/architecture.md)
- [Training Algorithm](../../autoresearch-master/program.md)
