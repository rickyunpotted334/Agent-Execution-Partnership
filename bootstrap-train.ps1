#!/usr/bin/env powershell
<#
.SYNOPSIS
    Bootstrap script for Agent Execution Partnership autoresearch training pipeline.

.DESCRIPTION
    Automates the full workflow:
      1. Install training dependencies (torch, rustbpe, tiktoken, etc.)
      2. Download dataset shards and train tokenizer
      3. Establish baseline BPB
      4. Run autonomous experiments (20 iterations)

.PARAMETER Depth
    Model depth (default: 8)

.PARAMETER Iterations
    Number of experiment iterations (default: 20)

.PARAMETER Shards
    Number of training shards to download (default: 100)

.PARAMETER SkipBaseline
    Skip baseline run and go straight to experiments

.EXAMPLE
    .\bootstrap-train.ps1
    .\bootstrap-train.ps1 -Depth 12 -Iterations 50 -Shards 200
    .\bootstrap-train.ps1 -SkipBaseline

.NOTES
    Requires: Python 3.12+, CUDA 12.1+, ~100GB storage for dataset
    Runtime: ~5 minutes baseline + ~100 minutes experiments (20x 5-min runs)
#>

param(
    [int]$Depth = 8,
    [int]$Iterations = 20,
    [int]$Shards = 100,
    [switch]$SkipBaseline = $false
)

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Text, [int]$Step, [int]$Total)
    Write-Host "[${Step}/${Total}] $Text" -ForegroundColor Green
}

function Invoke-Command-Checked {
    param([string]$Name, [scriptblock]$Command)
    Write-Step "Running: $Name" $script:CurrentStep $script:TotalSteps
    $script:CurrentStep++
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit 1
    }
}

$script:CurrentStep = 1
$script:TotalSteps = 5

Write-Header "AEP Autoresearch Training Bootstrap"
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Depth:        $Depth layers"
Write-Host "  Iterations:   $Iterations experiments"
Write-Host "  Shards:       $Shards training shards"
Write-Host "  Baseline:     $(if ($SkipBaseline) {'skipped'} else {'run first'})"
Write-Host ""

# Step 1: Install training deps
Invoke-Command-Checked "Install training dependencies" {
    py -m pip install -e ".[train]" --quiet
}

# Step 2: Prepare data
Invoke-Command-Checked "Download dataset and train tokenizer" {
    aep research prepare --shards $Shards
}

# Step 3: Establish baseline (unless skipped)
if (-not $SkipBaseline) {
    Invoke-Command-Checked "Establish baseline BPB" {
        aep research train --baseline --depth $Depth
    }
} else {
    Write-Step "SKIPPED: Baseline (use previous baseline from ledger)" $script:CurrentStep $script:TotalSteps
    $script:CurrentStep++
}

# Step 4: Run autonomous experiments
Invoke-Command-Checked "Run $Iterations autonomous experiments" {
    aep research train --depth $Depth --iterations $Iterations
}

Write-Header "✓ Training pipeline complete"
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  Experiment ledger: research/aee-autoresearch/experiment_ledger.jsonl"
Write-Host "  Results TSV:       research/aee-autoresearch/results.tsv"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review experiment_ledger.jsonl to analyze results"
Write-Host "  2. Check results.tsv for training metrics: BPB, memory, MFU"
Write-Host "  3. Retained experiments are new baselines for next run"
Write-Host ""
