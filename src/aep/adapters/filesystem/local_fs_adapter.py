from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from aep.config.settings import get_settings
from aep.contracts.models import ActionRequest


class FilesystemAdapter:
    def __init__(self) -> None:
        self.settings = get_settings()

    def execute(self, action: ActionRequest) -> dict[str, object]:
        target = Path(str(action.target.get("path", ""))).resolve()
        self._validate_target(target)
        op = action.operation

        if op == "read_file":
            return {
                "status": "success",
                "artifacts": [str(target)],
                "state_after": "read",
                "content": target.read_text(encoding="utf-8"),
                "logs_reference": "fs://read",
            }
        if op == "list_dir":
            return {
                "status": "success",
                "entries": sorted([p.name for p in target.iterdir()]),
                "state_after": "listed",
                "logs_reference": "fs://list",
            }
        if op == "create_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(action.arguments.get("content", "")), encoding="utf-8")
            return {"status": "success", "artifacts": [str(target)], "state_after": "created"}
        if op == "update_file":
            target.write_text(str(action.arguments.get("content", "")), encoding="utf-8")
            return {"status": "success", "artifacts": [str(target)], "state_after": "updated"}
        if op == "move_file":
            dest = Path(str(action.arguments.get("dest", ""))).resolve()
            self._validate_target(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(dest))
            return {"status": "success", "artifacts": [str(dest)], "state_after": "moved"}
        if op == "copy_file":
            dest = Path(str(action.arguments.get("dest", ""))).resolve()
            self._validate_target(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target), str(dest))
            return {"status": "success", "artifacts": [str(dest)], "state_after": "copied"}
        if op == "delete_file":
            if not self.settings.enable_destructive:
                return {"status": "failed", "error": "delete_disabled", "retryable": False}
            target.unlink(missing_ok=True)
            return {"status": "success", "state_after": "deleted"}
        if op == "checksum":
            payload = target.read_bytes()
            return {
                "status": "success",
                "state_after": "checksummed",
                "checksum_sha256": hashlib.sha256(payload).hexdigest(),
            }
        return {"status": "failed", "error": f"unsupported_operation:{op}"}

    def _validate_target(self, path: Path) -> None:
        roots = [p.resolve() for p in self.settings.fs_roots]
        for root in roots:
            if _is_relative_to(path, root):
                if path.is_symlink():
                    raise PermissionError("symlink_target_forbidden")
                return
        raise PermissionError("path_outside_allowed_roots")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
