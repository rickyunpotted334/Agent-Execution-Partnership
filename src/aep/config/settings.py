from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEP_", env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = "sqlite:///E:/Agent-Execution-Partnership/AI-Memory/memory.db"
    enable_destructive: bool = False
    allowed_fs_roots: str = (
        "E:/Agent-Execution-Partnership/agent-execution-partnership/examples/tmp"
    )
    http_allowlist: str = "localhost,127.0.0.1"
    policy_version: str = "2026-07-20"
    model_version: str = "functiongemma-baseline-v1"
    schema_version: str = "1.0.0"
    audit_file: str = (
        "E:/Agent-Execution-Partnership/agent-execution-partnership/examples/audit-ledger.jsonl"
    )
    risk_budget_default: float = Field(default=100.0, ge=0)

    @property
    def fs_roots(self) -> list[Path]:
        return [Path(p.strip()) for p in self.allowed_fs_roots.split(",") if p.strip()]

    @property
    def http_allow_domains(self) -> set[str]:
        return {d.strip().lower() for d in self.http_allowlist.split(",") if d.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
