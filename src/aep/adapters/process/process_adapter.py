from __future__ import annotations

import hashlib
import shlex
import subprocess  # nosec B404
from pathlib import Path

from aep.contracts.models import ActionRequest

ALLOWED_EXECUTABLES = {"cmd", "powershell", "python", "py", "echo"}


class ProcessAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        cmd = str(action.arguments.get("command", "")).strip()
        if not cmd:
            return {"status": "failed", "error": "missing_command", "retryable": False}
        argv = shlex.split(cmd, posix=False)
        exe = argv[0].lower().replace(".exe", "")
        if exe not in ALLOWED_EXECUTABLES:
            return {"status": "failed", "error": "executable_not_allowed", "retryable": False}

        cwd = Path(str(action.arguments.get("cwd", "."))).resolve()
        timeout_s = min(max(action.timeout_ms // 1000, 1), 30)
        proc = subprocess.run(  # nosec B603
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "status": "success" if proc.returncode == 0 else "failed",
            "state_after": "completed",
            "resource_usage": {"timeout_seconds": timeout_s},
            "artifacts": [],
            "logs_reference": "proc://local",
            "response": {
                "exit_code": proc.returncode,
                "stdout_digest": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
                "stderr_digest": hashlib.sha256(proc.stderr.encode("utf-8")).hexdigest(),
            },
            "error": None if proc.returncode == 0 else "non_zero_exit",
            "retryable": proc.returncode != 0,
            "compensation_available": False,
        }
