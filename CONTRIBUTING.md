# Contributing

## Development Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .[dev]
```

## Quality Gates

- ruff check .
- mypy src
- pytest
- bandit -q -r src

## Pull Requests

- Include tests for behavior changes.
- Keep security-sensitive changes small and well documented.
- Avoid enabling production-impact adapters by default.
