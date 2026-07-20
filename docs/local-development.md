# Local Development

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .[dev]
py -m aep.cli init --export-schemas
py -m aep.cli serve
```
