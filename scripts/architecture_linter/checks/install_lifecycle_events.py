"""Lifecycle event firing architecture guard."""

from __future__ import annotations

from scripts.architecture_linter.checks.install_deployment_shared import (
    _facts_for,
    _present,
    _summary,
)
from scripts.architecture_linter.facts import FactsProvider
from scripts.architecture_linter.models import Violation

GUARD_LIFECYCLE_EVENT_FIRING = "install-deployment-lifecycle-event-firing"

_OWNER = "src/apm_cli/core/lifecycle_scripts.py"
_OWNER_TOKENS = (
    "def fire_lifecycle_event(",
    "def packages_from_dependencies(",
    "def report_required_script_error(",
)
# Commands that fire lifecycle events, and the error chain that reports a
# failed required script. Each routes through the owner instead of building
# its own event, runner call, or failure message.
_CONSUMERS = {
    "src/apm_cli/install/service.py": ("fire_lifecycle_event(",),
    "src/apm_cli/commands/install.py": ("report_required_script_error(",),
    "src/apm_cli/commands/update.py": (
        "fire_lifecycle_event(",
        "report_required_script_error(",
    ),
    "src/apm_cli/commands/uninstall/cli.py": (
        "fire_lifecycle_event(",
        "report_required_script_error(",
    ),
    "src/apm_cli/commands/compile/cli.py": (
        "fire_lifecycle_event(",
        "report_required_script_error(",
    ),
}
_FORBIDDEN = ("LifecycleEvent.create(", "runner.fire(", "PackageInfo(name=dep.")


def check_lifecycle_event_firing(provider: FactsProvider) -> tuple[Violation, ...]:
    """Lifecycle events fire, and required failures render, through one owner."""
    rule_id = GUARD_LIFECYCLE_EVENT_FIRING
    findings: list[Violation] = []
    owner, failures = _facts_for(provider, _OWNER, rule_id)
    findings.extend(failures)
    if not failures:
        missing = [token for token in _OWNER_TOKENS if not _present(owner, token)]
        if missing:
            findings.append(
                _summary(rule_id, _OWNER, "Lifecycle owner is missing: " + ", ".join(missing))
            )
    for path, tokens in _CONSUMERS.items():
        facts, failures = _facts_for(provider, path, rule_id)
        if failures:
            findings.extend(failures)
            continue
        missing = [token for token in tokens if not _present(facts, token)]
        bypass = [token for token in _FORBIDDEN if _present(facts, token)]
        if missing:
            findings.append(
                _summary(
                    rule_id,
                    path,
                    "Lifecycle firing must route through core/lifecycle_scripts.py: missing "
                    + ", ".join(missing),
                )
            )
        if bypass:
            findings.append(
                _summary(
                    rule_id,
                    path,
                    "Lifecycle events are built or fired outside core/lifecycle_scripts.py: "
                    + ", ".join(bypass),
                )
            )
    return tuple(findings)
