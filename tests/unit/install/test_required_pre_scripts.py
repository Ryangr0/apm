"""A failed ``required: true`` pre-* script stops install, update and uninstall."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from apm_cli.cli import cli


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


def test_required_pre_update_failure_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trusted
) -> None:
    from apm_cli.commands.update import _fire_update_scripts

    project = tmp_path / "project"
    _write_project(project, "pre-update")
    package = MagicMock(package_path=project)
    package.get_apm_dependencies.return_value = []
    with pytest.raises(SystemExit) as exited:
        _fire_update_scripts(
            "pre-update",
            apm_package=package,
            scope=SimpleNamespace(value="project"),
            logger=None,
            verbose=False,
        )
    assert exited.value.code == 1


def test_required_pre_uninstall_failure_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trusted
) -> None:
    from apm_cli.commands.uninstall.cli import _fire_uninstall_scripts

    project = tmp_path / "project"
    _write_project(project, "pre-uninstall")
    logger = MagicMock()
    with pytest.raises(SystemExit) as exited:
        _fire_uninstall_scripts(
            "pre-uninstall",
            packages=["org/pkg"],
            scope=SimpleNamespace(value="project"),
            manifest_path=project / "apm.yml",
            logger=logger,
            verbose=False,
            deploy_root=project,
        )
    assert exited.value.code == 1
    assert "Required pre-uninstall script exited with code 5" in logger.error.call_args.args[0]
