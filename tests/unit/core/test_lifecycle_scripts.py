"""Unit tests for lifecycle script models, runner, and file-based discovery."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from apm_cli.core.lifecycle_scripts import (
    LIFECYCLE_EVENTS,
    SCRIPT_TYPES,
    LifecycleEvent,
    LifecycleScriptRunner,
    PackageInfo,
    RequiredScriptError,
    ScriptEntry,
    _entries_from_lifecycle_map,
    _get_policy_scripts_dir,
    build_runner_from_context,
    discover_scripts,
    fire_lifecycle_event,
    is_pre_event,
    packages_from_dependencies,
    parse_apm_yml_lifecycle,
    parse_project_script_file,
    parse_script_file,
    report_required_script_error,
)
from apm_cli.core.script_executors import CommandOutcome


def _write_yaml(path: Path, data: dict) -> Path:
    """Write *data* as YAML to *path* and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
    return path


class TestLifecycleEvent:
    def test_to_json_round_trips(self) -> None:
        event = LifecycleEvent(
            event="post-install",
            packages=[PackageInfo(name="org/repo", reference="v1")],
            scope="project",
            timestamp="2026-01-01T00:00:00Z",
            cli_version="0.14.0",
            working_directory="/tmp/project",
        )
        payload = json.loads(event.to_json())
        assert payload["event"] == "post-install"
        assert payload["packages"] == [{"name": "org/repo", "reference": "v1"}]

    @patch("apm_cli.version.get_version", return_value="0.14.1")
    def test_create_factory_fills_version_and_timestamp(self, _mock_ver: MagicMock) -> None:
        event = LifecycleEvent.create(event="pre-install", packages=[PackageInfo(name="a/b")])
        assert event.event == "pre-install"
        assert event.cli_version == "0.14.1"
        assert event.timestamp


class TestScriptEntry:
    def test_effective_command_prefers_bash_on_unix(self) -> None:
        script = ScriptEntry(
            script_type="command",
            event="post-install",
            bash="./bash.sh",
            command="echo fallback",
        )
        with patch("platform.system", return_value="Linux"):
            assert script.effective_command == "./bash.sh"

    def test_effective_command_prefers_command_on_windows(self) -> None:
        script = ScriptEntry(
            script_type="command",
            event="post-install",
            bash="./bash.sh",
            command="powershell -File x.ps1",
        )
        with patch("platform.system", return_value="Windows"):
            assert script.effective_command == "powershell -File x.ps1"

    def test_effective_timeout_defaults(self) -> None:
        assert ScriptEntry(script_type="http", event="post-install").effective_timeout == 10
        assert ScriptEntry(script_type="command", event="post-install").effective_timeout == 30


class TestParseScriptFile:
    def test_parses_valid_json_file(self, tmp_path: Path) -> None:
        script_file = tmp_path / "scripts.json"
        script_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "scripts": {
                        "post-install": [
                            {"type": "command", "bash": "echo done"},
                            {"type": "http", "url": "https://x.com/script"},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        entries = parse_script_file(script_file)
        assert len(entries) == 2
        assert entries[0].bash == "echo done"
        assert entries[1].url == "https://x.com/script"

    def test_rejects_wrong_version(self, tmp_path: Path) -> None:
        script_file = tmp_path / "scripts.json"
        script_file.write_text(
            json.dumps({"version": 99, "scripts": {"post-install": []}}),
            encoding="utf-8",
        )
        assert parse_script_file(script_file) == []


class TestParseApmYmlLifecycle:
    def test_parses_lifecycle_block(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "apm.yml",
            {
                "name": "demo",
                "lifecycle": {
                    "post-install": [{"type": "command", "bash": "echo done"}],
                    "pre-uninstall": [{"type": "http", "url": "https://example.com/hook"}],
                },
            },
        )
        entries = parse_apm_yml_lifecycle(path, "project")
        assert len(entries) == 2
        assert entries[0].source == "project"
        assert entries[1].script_type == "http"

    def test_run_alias_populates_bash_and_command(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "run": "echo alias"}]}},
        )
        entry = parse_apm_yml_lifecycle(path, "project")[0]
        assert entry.bash == "echo alias"
        assert entry.command == "echo alias"

    def test_entries_from_lifecycle_map_needs_no_version(self, tmp_path: Path) -> None:
        entries = _entries_from_lifecycle_map(
            {"post-install": [{"type": "command", "bash": "echo ok"}]},
            tmp_path / "apm.yml",
            "project",
        )
        assert len(entries) == 1
        assert entries[0].event == "post-install"

    def test_parse_project_script_file_is_alias(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo ok"}]}},
        )
        assert parse_project_script_file(path) == parse_apm_yml_lifecycle(path, "project")


class TestRequiredField:
    def _entry(self, tmp_path: Path, event: str, raw: dict) -> ScriptEntry:
        path = _write_yaml(tmp_path / "apm.yml", {"lifecycle": {event: [raw]}})
        return parse_apm_yml_lifecycle(path, "project")[0]

    def test_required_true_on_pre_event_command(self, tmp_path: Path) -> None:
        entry = self._entry(tmp_path, "pre-compile", {"run": "gen", "required": True})
        assert entry.required is True
        assert entry.required_issue is None

    def test_required_defaults_to_false(self, tmp_path: Path) -> None:
        entry = self._entry(tmp_path, "pre-compile", {"run": "gen"})
        assert entry.required is False
        assert entry.required_issue is None

    @pytest.mark.parametrize("value", ["true", 1, "yes"])
    def test_non_boolean_is_an_issue(self, tmp_path: Path, value: object) -> None:
        entry = self._entry(tmp_path, "pre-compile", {"run": "gen", "required": value})
        assert entry.required is False
        assert entry.required_issue == "'required' must be true or false"

    def test_post_event_is_an_issue(self, tmp_path: Path) -> None:
        entry = self._entry(tmp_path, "post-install", {"run": "x", "required": True})
        assert entry.required is False
        assert entry.required_issue == "'required' applies only to pre-* events"

    def test_http_script_is_an_issue(self, tmp_path: Path) -> None:
        entry = self._entry(
            tmp_path, "pre-install", {"url": "https://example.com/hook", "required": True}
        )
        assert entry.required is False
        assert entry.required_issue == "'required' applies only to command scripts"

    @pytest.mark.parametrize(
        ("event", "expected"),
        [("pre-compile", True), ("pre-install", True), ("post-install", False), ("pre-x", False)],
    )
    def test_is_pre_event(self, event: str, expected: bool) -> None:
        assert is_pre_event(event) is expected


@pytest.mark.windows_compat
class TestGetPolicyScriptsDir:
    def test_windows_honours_programdata_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setenv("PROGRAMDATA", r"D:\ProgramData")
        assert _get_policy_scripts_dir() == Path(r"D:\ProgramData") / "APM" / "policy.d"

    def test_windows_falls_back_to_c_drive_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.delenv("PROGRAMDATA", raising=False)
        assert _get_policy_scripts_dir() == Path(r"C:\ProgramData") / "APM" / "policy.d"

    @pytest.mark.parametrize("program_data", ["", "relative"])
    def test_windows_falls_back_when_programdata_is_unsafe(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program_data: str,
    ) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setenv("PROGRAMDATA", program_data)
        assert _get_policy_scripts_dir() == Path(r"C:\ProgramData") / "APM" / "policy.d"

    def test_windows_discovers_policy_scripts_from_programdata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if os.name == "nt":
            program_data = tmp_path / "ProgramData"
            program_data_value = str(program_data)
        else:
            program_data_value = r"D:\ProgramData"
            program_data = tmp_path / program_data_value
            monkeypatch.chdir(tmp_path)
        script_file = program_data / "APM" / "policy.d" / "admin.json"
        script_file.parent.mkdir(parents=True)
        script_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "scripts": {"post-install": [{"type": "command", "command": "echo admin"}]},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setenv("PROGRAMDATA", program_data_value)

        entries = discover_scripts(project_root=str(tmp_path / "project"))

        assert [(entry.source, entry.command) for entry in entries] == [("policy", "echo admin")]

    def test_non_windows_uses_etc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Linux")
        assert _get_policy_scripts_dir() == Path("/etc/apm/policy.d")


class TestDiscoverScripts:
    def test_discovers_from_project_file(self, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo project"}]}},
        )
        entries = discover_scripts(project_root=str(tmp_path))
        assert any(entry.bash == "echo project" for entry in entries)

    def test_discovers_from_user_apm_yml(self, tmp_path: Path) -> None:
        user_apm = _write_yaml(
            tmp_path / "user" / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo user"}]}},
        )
        with patch("apm_cli.core.lifecycle_scripts._get_user_apm_yml", return_value=user_apm):
            entries = discover_scripts()
        assert any(entry.bash == "echo user" for entry in entries)

    def test_additive_across_sources(self, tmp_path: Path) -> None:
        user_apm = _write_yaml(
            tmp_path / "user" / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo user"}]}},
        )
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        _write_yaml(
            project_dir / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo project"}]}},
        )
        with patch("apm_cli.core.lifecycle_scripts._get_user_apm_yml", return_value=user_apm):
            entries = discover_scripts(project_root=str(project_dir))
        assert len(entries) == 2


class TestLifecycleScriptRunner:
    def _make_event(self, event_name: str = "post-install") -> LifecycleEvent:
        return LifecycleEvent(
            event=event_name,
            packages=[PackageInfo(name="org/repo")],
            scope="project",
            timestamp="2026-01-01T00:00:00Z",
            cli_version="0.0.0",
            working_directory="/tmp/test",
        )

    def test_fire_calls_matching_scripts(self) -> None:
        script = ScriptEntry(script_type="command", event="post-install", bash="echo hi")
        runner = LifecycleScriptRunner(scripts=[script])
        with patch("apm_cli.core.script_executors.execute_script") as mock_exec:
            runner.fire("post-install", self._make_event())
        mock_exec.assert_called_once()

    def test_error_isolation_does_not_block_other_scripts(self) -> None:
        script1 = ScriptEntry(script_type="command", event="post-install", bash="fail")
        script2 = ScriptEntry(script_type="command", event="post-install", bash="ok")
        runner = LifecycleScriptRunner(scripts=[script1, script2])
        call_count = 0

        def _side_effect(script, event, **kw):
            nonlocal call_count
            call_count += 1
            if script.bash == "fail":
                raise RuntimeError("boom")

        with patch("apm_cli.core.script_executors.execute_script", side_effect=_side_effect):
            runner.fire("post-install", self._make_event())
        assert call_count == 2

    def _required(self, **kw) -> ScriptEntry:
        return ScriptEntry(
            script_type="command", event="pre-compile", bash="gen", required=True, **kw
        )

    def test_required_failure_raises_and_stops_later_scripts(self) -> None:
        later = ScriptEntry(script_type="command", event="pre-compile", bash="later")
        runner = LifecycleScriptRunner(scripts=[self._required(), later])
        failed = CommandOutcome(status="error", exit_code=3, stderr_lines=("no composer.json",))
        with (
            patch("apm_cli.core.script_executors.run_command_script", return_value=failed),
            patch("apm_cli.core.script_executors.execute_script") as mock_exec,
            pytest.raises(RequiredScriptError) as raised,
        ):
            runner.fire("pre-compile", self._make_event("pre-compile"))
        assert str(raised.value) == "Required pre-compile script exited with code 3: gen"
        assert raised.value.detail == ("no composer.json",)
        mock_exec.assert_not_called()

    def test_required_failure_label_names_description_and_command(self) -> None:
        runner = LifecycleScriptRunner(scripts=[self._required(description="Generate rules")])
        with (
            patch(
                "apm_cli.core.script_executors.run_command_script",
                return_value=CommandOutcome(status="timeout"),
            ),
            pytest.raises(RequiredScriptError) as raised,
        ):
            runner.fire("pre-compile", self._make_event("pre-compile"))
        assert str(raised.value) == (
            "Required pre-compile script timed out after 30s: Generate rules (gen)"
        )

    def test_required_success_continues(self) -> None:
        later = ScriptEntry(script_type="command", event="pre-compile", bash="later")
        runner = LifecycleScriptRunner(scripts=[self._required(), later])
        with (
            patch(
                "apm_cli.core.script_executors.run_command_script",
                return_value=CommandOutcome(status="ok", exit_code=0),
            ),
            patch("apm_cli.core.script_executors.execute_script") as mock_exec,
        ):
            runner.fire("pre-compile", self._make_event("pre-compile"))
        mock_exec.assert_called_once()

    def test_real_required_script_failure_raises(self, tmp_path: Path) -> None:
        command = f'"{sys.executable}" -c "import sys; sys.exit(6)"'
        script = ScriptEntry(
            script_type="command", event="pre-compile", command=command, required=True
        )
        runner = LifecycleScriptRunner(scripts=[script], project_root=str(tmp_path))
        with pytest.raises(RequiredScriptError, match="exited with code 6"):
            runner.fire("pre-compile", self._make_event("pre-compile"))

    def test_untrusted_required_script_raises_without_skip_notice(self) -> None:
        logger = MagicMock()
        runner = LifecycleScriptRunner(
            scripts=[],
            logger=logger,
            skipped_project_scripts=1,
            skipped_project_file="/repo/apm.yml",
            skipped_entries=[self._required()],
        )
        with pytest.raises(RequiredScriptError, match="apm lifecycle trust"):
            runner.fire("pre-compile", self._make_event("pre-compile"))
        logger.warning.assert_not_called()

    def test_untrusted_required_script_for_other_event_does_not_raise(self) -> None:
        skipped = ScriptEntry(script_type="command", event="pre-install", bash="g", required=True)
        runner = LifecycleScriptRunner(scripts=[], skipped_entries=[skipped])
        assert runner.fire("pre-compile", self._make_event("pre-compile")) == []

    def test_skip_notice_counts_only_this_event(self) -> None:
        logger = MagicMock()
        skipped = [
            ScriptEntry(script_type="command", event="pre-compile", bash="a"),
            ScriptEntry(script_type="command", event="post-install", bash="b"),
            ScriptEntry(script_type="command", event="post-install", bash="c"),
        ]
        runner = LifecycleScriptRunner(
            scripts=[], logger=logger, skipped_project_scripts=3, skipped_entries=skipped
        )
        runner.fire("pre-compile", self._make_event("pre-compile"))
        logger.warning.assert_called_once_with(
            "Skipped 1 untrusted project script(s) for pre-compile. "
            "Run 'apm lifecycle trust' to enable them."
        )

    def test_invalid_required_value_on_pre_command_raises(self) -> None:
        script = ScriptEntry(
            script_type="command",
            event="pre-compile",
            bash="gen",
            required_issue="'required' must be true or false",
            source_file="apm.yml",
        )
        runner = LifecycleScriptRunner(scripts=[script])
        with (
            patch("apm_cli.core.script_executors.execute_script") as mock_exec,
            pytest.raises(RequiredScriptError, match="must be true or false"),
        ):
            runner.fire("pre-compile", self._make_event("pre-compile"))
        mock_exec.assert_not_called()

    def test_required_on_post_event_warns_and_runs(self) -> None:
        logger = MagicMock()
        script = ScriptEntry(
            script_type="command",
            event="post-install",
            bash="x",
            required_issue="'required' applies only to pre-* events",
            source_file="apm.yml",
        )
        runner = LifecycleScriptRunner(scripts=[script], logger=logger)
        with patch("apm_cli.core.script_executors.execute_script") as mock_exec:
            runner.fire("post-install", self._make_event())
        mock_exec.assert_called_once()
        logger.warning.assert_called_once_with(
            "post-install script in apm.yml: 'required' applies only to pre-* events; ignored."
        )

    def test_scripts_for_event_filters(self) -> None:
        s1 = ScriptEntry(script_type="command", event="post-install", bash="echo a")
        s2 = ScriptEntry(script_type="http", event="pre-install", url="https://x.com")
        runner = LifecycleScriptRunner(scripts=[s1, s2])
        assert runner.scripts_for_event("post-install") == [s1]


class TestBuildRunnerFromContext:
    def test_apm_no_scripts_env_returns_empty_runner(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("APM_NO_SCRIPTS", "1")
        runner = build_runner_from_context(project_root=str(tmp_path))
        assert runner._scripts == []

    def test_deny_all_true_suppresses_all_scripts(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo hi"}]}},
        )
        mock_policy = MagicMock()
        mock_policy.executables.deny_all = True
        mock_fetch = MagicMock()
        mock_fetch.policy = mock_policy
        with patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=mock_fetch):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert runner._scripts == []

    def test_deny_all_false_does_not_suppress_user_scripts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        mock_policy = MagicMock()
        mock_policy.executables.deny_all = False
        mock_fetch = MagicMock()
        mock_fetch.policy = mock_policy
        user_apm = _write_yaml(
            tmp_path / "user" / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo user"}]}},
        )
        with (
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=mock_fetch),
            patch("apm_cli.core.lifecycle_scripts._get_user_apm_yml", return_value=user_apm),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert any(script.bash == "echo user" for script in runner._scripts)

    def test_untrusted_project_scripts_are_skipped(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo from-project"}]}},
        )
        with (
            patch("apm_cli.core.script_trust.is_fingerprint_trusted", return_value=False),
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=None),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert runner._scripts == []
        assert runner._skipped_project_scripts == 1

    def test_untrusted_required_project_script_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"pre-compile": [{"run": "gen", "required": True}]}},
        )
        with (
            patch("apm_cli.core.script_trust.is_fingerprint_trusted", return_value=False),
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=None),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        with pytest.raises(RequiredScriptError):
            runner.fire("pre-compile", LifecycleEvent.create(event="pre-compile"))

    def test_apm_no_scripts_skips_required_scripts(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("APM_NO_SCRIPTS", "1")
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"pre-compile": [{"run": "gen", "required": True}]}},
        )
        runner = build_runner_from_context(project_root=str(tmp_path))
        assert runner.fire("pre-compile", LifecycleEvent.create(event="pre-compile")) == []

    def test_deny_all_skips_untrusted_required_scripts(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"pre-compile": [{"run": "gen", "required": True}]}},
        )
        fetch = MagicMock()
        fetch.policy.executables.deny_all = True
        with (
            patch("apm_cli.core.script_trust.is_fingerprint_trusted", return_value=False),
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=fetch),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert runner.fire("pre-compile", LifecycleEvent.create(event="pre-compile")) == []

    def test_events_filter_skips_policy_discovery(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"run": "echo hi"}]}},
        )
        with patch("apm_cli.policy.discovery.discover_policy_with_chain") as discover:
            runner = build_runner_from_context(project_root=str(tmp_path), events=("pre-compile",))
        discover.assert_not_called()
        assert runner.scripts_for_event("post-install") == []

    def test_trusted_project_scripts_are_included(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        _write_yaml(
            tmp_path / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo trusted"}]}},
        )
        with (
            patch("apm_cli.core.script_trust.is_fingerprint_trusted", return_value=True),
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=None),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert any(script.bash == "echo trusted" for script in runner._scripts)
        assert runner._skipped_project_scripts == 0

    def test_user_scripts_bypass_trust_gate(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("APM_NO_SCRIPTS", raising=False)
        user_apm = _write_yaml(
            tmp_path / "user" / "apm.yml",
            {"lifecycle": {"post-install": [{"type": "command", "bash": "echo user-notify"}]}},
        )
        with (
            patch("apm_cli.core.lifecycle_scripts._get_user_apm_yml", return_value=user_apm),
            patch("apm_cli.policy.discovery.discover_policy_with_chain", return_value=None),
        ):
            runner = build_runner_from_context(project_root=str(tmp_path))
        assert any(script.bash == "echo user-notify" for script in runner._scripts)


class TestFireLifecycleEvent:
    def test_fires_through_supplied_runner(self) -> None:
        runner = MagicMock()
        runner.fire.return_value = []
        fire_lifecycle_event(
            "pre-install",
            packages=packages_from_dependencies([]),
            scope="project",
            project_root="/repo",
            runner=runner,
        )
        assert runner.fire.call_args.args[0] == "pre-install"
        assert runner.fire.call_args.args[1].event == "pre-install"

    def test_isolates_unexpected_errors(self) -> None:
        runner = MagicMock()
        runner.fire.side_effect = RuntimeError("boom")
        assert (
            fire_lifecycle_event(
                "post-install", packages=[], scope="project", project_root=None, runner=runner
            )
            == []
        )

    def test_reraises_required_script_error(self) -> None:
        runner = MagicMock()
        runner.fire.side_effect = RequiredScriptError("Required pre-install script failed")
        with pytest.raises(RequiredScriptError):
            fire_lifecycle_event(
                "pre-install", packages=[], scope="project", project_root=None, runner=runner
            )

    def test_packages_from_dependencies(self) -> None:
        dep = MagicMock(repo_url="org/repo", reference="v1")
        assert packages_from_dependencies([dep]) == [PackageInfo(name="org/repo", reference="v1")]


class TestReportRequiredScriptError:
    def test_quiet_points_at_verbose_and_log(self) -> None:
        logger = MagicMock()
        error = RequiredScriptError("Required pre-compile script exited with code 1: gen", ("x",))
        report_required_script_error(error, logger, verbose=False)
        logger.error.assert_called_once_with(str(error))
        assert "--verbose" in logger.progress.call_args.args[0]
        logger.verbose_detail.assert_not_called()

    def test_verbose_shows_indented_detail(self) -> None:
        logger = MagicMock()
        error = RequiredScriptError("Required pre-compile script failed: gen", ("a", "b"))
        report_required_script_error(error, logger, verbose=True)
        assert [c.args[0] for c in logger.verbose_detail.call_args_list] == ["    a", "    b"]


class TestConstants:
    def test_lifecycle_events_tuple(self) -> None:
        assert set(LIFECYCLE_EVENTS) == {
            "pre-install",
            "post-install",
            "pre-update",
            "post-update",
            "pre-uninstall",
            "post-uninstall",
            "pre-compile",
        }

    def test_script_types_tuple(self) -> None:
        assert set(SCRIPT_TYPES) == {"command", "http"}
