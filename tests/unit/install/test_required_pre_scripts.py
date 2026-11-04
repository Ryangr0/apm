"""A failed ``required: true`` pre-install script stops ``apm install``.

Update and uninstall are covered by the real-CLI state test in
``tests/integration/test_required_lifecycle_state_machine.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from apm_cli.cli import cli

pytestmark = [pytest.mark.component, pytest.mark.windows_compat]


def _write_project(project: Path, event: str) -> None:
    entry = {
        "type": "command",
        "description": "Generate sources",
        "run": f'"{sys.executable}" -c "import sys; sys.exit(5)"',
        "required": True,
    }
    project.mkdir(parents=True, exist_ok=True)
    (project / "apm.yml").write_text(
        yaml.safe_dump({"name": "demo", "version": "0.1.0", "lifecycle": {event: [entry]}})
    )


@pytest.fixture
def trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APM_HOME", str(tmp_path / "apmhome"))
    monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
    with (
        patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=None),
        patch("apm_cli.core.script_trust.is_fingerprint_trusted", return_value=True),
    ):
        yield


def test_required_pre_install_failure_fails_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trusted
) -> None:
    project = tmp_path / "project"
    _write_project(project, "pre-install")
    instructions = project / ".apm" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "kept.instructions.md").write_text('---\napplyTo: "**"\n---\nKept.\n')
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli, ["install"])
    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    assert "[x] Required pre-install script exited with code 5: Generate sources" in output
    assert "Failed to install APM dependencies" not in output
