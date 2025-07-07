from typing import Any

from pydantic import BaseModel

from openhands.resolver.interfaces.issue import Issue


class PlanOutput(BaseModel):
    # NOTE: User-specified
    issue: Issue
    issue_type: str
    instruction: str
    base_commit: str
    plan: str  # The actual plan text
    history: list[dict[str, Any]]
    metrics: dict[str, Any] | None
    success: bool
    error: str | None
