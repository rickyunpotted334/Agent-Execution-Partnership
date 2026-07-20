from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterator

from aep.contracts.models import Observation


class ObservationService:
    """In-process store for observations with freshness enforcement."""

    def __init__(self) -> None:
        self._store: dict[str, Observation] = {}

    def record(self, observation: Observation) -> Observation:
        """Persist an observation and return it."""
        self._store[observation.observation_id] = observation
        return observation

    def get(self, observation_id: str) -> Observation | None:
        return self._store.get(observation_id)

    def list_for_task(self, task_id: str) -> list[Observation]:
        return [o for o in self._store.values() if o.task_id == task_id]

    def is_fresh(self, observation: Observation) -> bool:
        """Return True if the observation's freshness deadline has not passed."""
        return observation.freshness_deadline >= datetime.now(UTC)

    def get_by_state_version(self, task_id: str, state_version: str) -> Observation | None:
        for obs in self._store.values():
            if obs.task_id == task_id and obs.state_version == state_version:
                return obs
        return None

    def latest_fresh(self, task_id: str) -> Observation | None:
        """Return the most-recent fresh observation for a task, or None."""
        candidates = [o for o in self.list_for_task(task_id) if self.is_fresh(o)]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.timestamp)

    def purge_stale(self) -> int:
        """Remove all stale observations. Returns the count removed."""
        now = datetime.now(UTC)
        stale = [oid for oid, o in self._store.items() if o.freshness_deadline < now]
        for oid in stale:
            del self._store[oid]
        return len(stale)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)
