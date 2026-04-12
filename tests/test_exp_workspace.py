"""Tests for tools/exp_workspace.py — Experiment workspace manager."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.exp_workspace import (
    ALL_PHASES,
    REQUIRED_DIRS,
    REQUIRED_FILES,
    SCAFFOLD_TYPES,
    VALID_TRANSITIONS,
    WORKSPACE_BASE,
    _deep_merge,
    _dump_yaml_simple,
    _find_project_root,
    _format_scalar,
    _read_experiment_yaml,
    _workspace_path,
    _write_experiment_yaml,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / "CLAUDE.md").write_text("# test", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / WORKSPACE_BASE).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def workspace(tmp_project):
    """Create a workspace with .experiment.yaml."""
    slug = "test-exp"
    ws = tmp_project / WORKSPACE_BASE / slug
    ws.mkdir(parents=True)
    for d in REQUIRED_DIRS:
        (ws / d).mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_FILES:
        fp = ws / f
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(f"# placeholder\nprint('hello')\n", encoding="utf-8")
    # Write .experiment.yaml
    yaml = textwrap.dedent("""\
        slug: test-exp
        claim: test-claim
        idea: test-idea
        type: validation
        workspace_state:
          phase: designed
          phases_completed: {}
          fix_rounds_total: 0
          blocked: false
          blocked_reason: ""
    """)
    (ws / ".experiment.yaml").write_text(yaml, encoding="utf-8")
    # Trajectory
    traj = ws / ".trajectory"
    traj.mkdir(exist_ok=True)
    (traj / "events.jsonl").touch()
    (traj / "fixes.jsonl").touch()
    return ws


def _run_tool(tmp_project, *cli_args):
    """Run exp_workspace.py as subprocess and return parsed JSON."""
    cmd = [sys.executable, str(ROOT / "tools" / "exp_workspace.py")] + list(cli_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_project))
    try:
        return json.loads(proc.stdout), proc.returncode
    except json.JSONDecodeError:
        return {"raw": proc.stdout, "stderr": proc.stderr}, proc.returncode


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


class TestYamlHelpers:
    def test_format_scalar_string(self):
        assert _format_scalar("hello") == "hello"

    def test_format_scalar_empty(self):
        assert _format_scalar("") == '""'

    def test_format_scalar_bool(self):
        assert _format_scalar(True) == "true"
        assert _format_scalar(False) == "false"

    def test_format_scalar_number(self):
        assert _format_scalar(42) == "42"
        assert _format_scalar(3.14) == "3.14"

    def test_format_scalar_special_chars(self):
        result = _format_scalar("hello: world")
        assert result.startswith('"')

    def test_dump_yaml_simple_basic(self):
        data = {"key": "value", "num": 42}
        result = _dump_yaml_simple(data)
        assert "key: value" in result
        assert "num: 42" in result

    def test_dump_yaml_nested(self):
        data = {"outer": {"inner": "value"}}
        result = _dump_yaml_simple(data)
        assert "outer:" in result
        assert "inner: value" in result

    def test_dump_yaml_list(self):
        data = {"items": [1, 2, 3]}
        result = _dump_yaml_simple(data)
        assert "[1, 2, 3]" in result

    def test_dump_yaml_empty_dict(self):
        data = {"empty": {}}
        result = _dump_yaml_simple(data)
        assert "empty: {}" in result

    def test_dump_yaml_empty_list(self):
        data = {"empty": []}
        result = _dump_yaml_simple(data)
        assert "empty: []" in result


class TestDeepMerge:
    def test_simple_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_replace_dict_with_scalar(self):
        assert _deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


class TestScaffold:
    def test_scaffold_creates_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "scaffold", "--slug", "new-exp")
        assert rc == 0
        assert data["status"] == "ok"
        assert "new-exp" in data["slug"]
        ws = tmp_project / WORKSPACE_BASE / "new-exp"
        assert ws.exists()

    def test_scaffold_creates_required_dirs(self, tmp_project):
        _run_tool(tmp_project, "scaffold", "--slug", "dir-test")
        ws = tmp_project / WORKSPACE_BASE / "dir-test"
        for d in REQUIRED_DIRS:
            assert (ws / d).is_dir(), f"Missing dir: {d}"

    def test_scaffold_creates_required_files(self, tmp_project):
        _run_tool(tmp_project, "scaffold", "--slug", "file-test")
        ws = tmp_project / WORKSPACE_BASE / "file-test"
        for f in REQUIRED_FILES:
            assert (ws / f).exists(), f"Missing file: {f}"

    def test_scaffold_creates_experiment_yaml(self, tmp_project):
        _run_tool(tmp_project, "scaffold", "--slug", "yaml-test")
        ws = tmp_project / WORKSPACE_BASE / "yaml-test"
        assert (ws / ".experiment.yaml").exists()

    def test_scaffold_creates_trajectory(self, tmp_project):
        _run_tool(tmp_project, "scaffold", "--slug", "traj-test")
        ws = tmp_project / WORKSPACE_BASE / "traj-test"
        assert (ws / ".trajectory" / "events.jsonl").exists()
        assert (ws / ".trajectory" / "fixes.jsonl").exists()

    def test_scaffold_with_type(self, tmp_project):
        data, rc = _run_tool(tmp_project, "scaffold", "--slug", "cls-exp", "--type", "classification")
        assert rc == 0
        ws = tmp_project / WORKSPACE_BASE / "cls-exp"
        assert (ws / "src" / "model.py").exists()
        assert (ws / "src" / "dataset.py").exists()

    def test_scaffold_fails_if_exists(self, tmp_project):
        _run_tool(tmp_project, "scaffold", "--slug", "dup-test")
        data, rc = _run_tool(tmp_project, "scaffold", "--slug", "dup-test")
        assert rc != 0
        assert data["status"] == "error"

    def test_scaffold_returns_file_list(self, tmp_project):
        data, rc = _run_tool(tmp_project, "scaffold", "--slug", "list-test")
        assert rc == 0
        assert "created_dirs" in data
        assert "created_files" in data
        assert len(data["created_dirs"]) >= len(REQUIRED_DIRS)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_valid_workspace(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "validate", "--slug", "test-exp")
        assert rc == 0
        assert data["valid"] is True
        assert data["issues"] == []

    def test_validate_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "validate", "--slug", "nonexistent")
        assert rc != 0

    def test_validate_missing_dir(self, tmp_project, workspace):
        import shutil
        shutil.rmtree(workspace / "src")
        data, rc = _run_tool(tmp_project, "validate", "--slug", "test-exp")
        assert rc == 0
        assert data["valid"] is False
        assert any(i["path"] == "src" for i in data["issues"])

    def test_validate_missing_file(self, tmp_project, workspace):
        (workspace / "requirements.txt").unlink()
        data, rc = _run_tool(tmp_project, "validate", "--slug", "test-exp")
        assert rc == 0
        assert data["valid"] is False

    def test_validate_syntax_error(self, tmp_project, workspace):
        (workspace / "src" / "bad.py").write_text("def broken(\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "validate", "--slug", "test-exp")
        assert rc == 0
        assert any(i["type"] == "syntax_error" for i in data["issues"])

    def test_validate_empty_file_warning(self, tmp_project, workspace):
        (workspace / "src" / "train.py").write_text("", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "validate", "--slug", "test-exp")
        assert rc == 0
        assert any(w["type"] == "empty_file" for w in data["warnings"])


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_initializes_git(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "init")
        assert rc == 0
        assert data["committed"] is True
        assert (workspace / ".git").exists()

    def test_snapshot_creates_tag(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "v1")
        proc = subprocess.run(
            ["git", "tag", "-l"], cwd=workspace, capture_output=True, text=True,
        )
        assert "v1" in proc.stdout

    def test_snapshot_no_changes(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "first")
        data, rc = _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "second")
        assert rc == 0
        assert data["committed"] is False

    def test_snapshot_with_message(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "msg-test", "--message", "custom msg")
        assert rc == 0
        assert data["message"] == "custom msg"

    def test_snapshot_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "snapshot", "--slug", "nope", "--tag", "x")
        assert rc != 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_after_snapshot(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "base")
        (workspace / "src" / "new.py").write_text("print('hi')\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "diff", "--slug", "test-exp")
        assert rc == 0
        assert data["has_changes"] is True
        assert "src/new.py" in data["untracked_files"]

    def test_diff_no_changes(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "base")
        data, rc = _run_tool(tmp_project, "diff", "--slug", "test-exp")
        assert rc == 0
        assert data["has_changes"] is False

    def test_diff_no_git(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "diff", "--slug", "test-exp")
        assert rc != 0

    def test_diff_modified_file(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "base")
        (workspace / "src" / "train.py").write_text("print('modified')\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "diff", "--slug", "test-exp")
        assert rc == 0
        assert "src/train.py" in data["changed_files"]

    def test_diff_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "diff", "--slug", "nope")
        assert rc != 0


# ---------------------------------------------------------------------------
# config-compose
# ---------------------------------------------------------------------------


class TestConfigCompose:
    def test_compose_single_file(self, tmp_project, workspace):
        (workspace / "configs" / "base.yaml").write_text("lr: 0.001\nbatch_size: 32\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "config-compose", "--slug", "test-exp", "--base", "configs/base.yaml")
        assert rc == 0
        assert data["composed"]["lr"] == 0.001

    def test_compose_with_override(self, tmp_project, workspace):
        (workspace / "configs" / "base.yaml").write_text("lr: 0.001\nbatch_size: 32\n", encoding="utf-8")
        (workspace / "configs" / "big.yaml").write_text("batch_size: 128\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "config-compose", "--slug", "test-exp", "--base", "configs/base.yaml", "--override", "configs/big.yaml")
        assert rc == 0
        assert data["composed"]["batch_size"] == 128
        assert data["composed"]["lr"] == 0.001

    def test_compose_missing_base(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "config-compose", "--slug", "test-exp", "--base", "nonexistent.yaml")
        assert rc != 0

    def test_compose_missing_override(self, tmp_project, workspace):
        (workspace / "configs" / "base.yaml").write_text("lr: 0.001\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "config-compose", "--slug", "test-exp", "--base", "configs/base.yaml", "--override", "nonexistent.yaml")
        assert rc != 0

    def test_compose_from_preset(self, tmp_project, workspace):
        # Create a preset in config/experiment-templates/
        templates_dir = tmp_project / "config" / "experiment-templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        (templates_dir / "cls.yaml").write_text("type: classification\nmetrics: [accuracy]\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "config-compose", "--slug", "test-exp", "--base", "cls.yaml")
        assert rc == 0
        assert data["composed"]["type"] == "classification"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_basic(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "status", "--slug", "test-exp")
        assert rc == 0
        assert data["phase"] == "designed"
        assert data["python_files"] >= 1
        assert data["total_files"] >= 1

    def test_status_shows_blocked(self, tmp_project, workspace):
        cfg = _read_experiment_yaml(workspace)
        cfg["workspace_state"]["blocked"] = True
        _write_experiment_yaml(workspace, cfg)
        data, rc = _run_tool(tmp_project, "status", "--slug", "test-exp")
        assert rc == 0
        assert data["blocked"] is True

    def test_status_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "status", "--slug", "nope")
        assert rc != 0

    def test_status_git_info(self, tmp_project, workspace):
        _run_tool(tmp_project, "snapshot", "--slug", "test-exp", "--tag", "init")
        data, rc = _run_tool(tmp_project, "status", "--slug", "test-exp")
        assert rc == 0
        assert data["git_initialized"] is True

    def test_status_trajectory_counts(self, tmp_project, workspace):
        (workspace / ".trajectory" / "events.jsonl").write_text('{"event":"test"}\n', encoding="utf-8")
        data, rc = _run_tool(tmp_project, "status", "--slug", "test-exp")
        assert rc == 0
        assert data["events_logged"] == 1


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class TestState:
    def test_state_reads_phase(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "state", "--slug", "test-exp")
        assert rc == 0
        assert data["workspace_state"]["phase"] == "designed"

    def test_state_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "state", "--slug", "nope")
        assert rc != 0

    def test_state_missing_yaml(self, tmp_project):
        slug = "no-yaml"
        ws = tmp_project / WORKSPACE_BASE / slug
        ws.mkdir(parents=True)
        data, rc = _run_tool(tmp_project, "state", "--slug", slug)
        assert rc != 0

    def test_state_returns_full_state(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "state", "--slug", "test-exp")
        assert rc == 0
        ws_state = data["workspace_state"]
        assert "phase" in ws_state
        assert "blocked" in ws_state

    def test_state_slug_in_output(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "state", "--slug", "test-exp")
        assert rc == 0
        assert data["slug"] == "test-exp"


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_valid_transition(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "scaffolded")
        assert rc == 0
        assert data["previous_phase"] == "designed"
        assert data["new_phase"] == "scaffolded"

    def test_invalid_transition(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "running")
        assert rc != 0

    def test_transition_to_blocked(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "blocked", "--reason", "test failure")
        assert rc == 0
        # Verify state
        state_data, _ = _run_tool(tmp_project, "state", "--slug", "test-exp")
        assert state_data["workspace_state"]["blocked"] is True

    def test_transition_chain(self, tmp_project, workspace):
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "scaffolded")
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "validated")
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "running")
        assert rc == 0
        assert data["new_phase"] == "running"

    def test_transition_records_timestamp(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "scaffolded")
        assert rc == 0
        assert "timestamp" in data

    def test_transition_invalid_phase(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "nonexistent")
        assert rc != 0

    def test_transition_with_reason(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "scaffolded", "--reason", "code generated")
        assert rc == 0
        assert data["reason"] == "code generated"

    def test_transition_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "transition", "--slug", "nope", "--to", "scaffolded")
        assert rc != 0


# ---------------------------------------------------------------------------
# unblock
# ---------------------------------------------------------------------------


class TestUnblock:
    def test_unblock_reverts_phase(self, tmp_project, workspace):
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "scaffolded")
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "blocked", "--reason", "error")
        data, rc = _run_tool(tmp_project, "unblock", "--slug", "test-exp")
        assert rc == 0
        assert data["reverted_to"] == "scaffolded"

    def test_unblock_not_blocked(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "unblock", "--slug", "test-exp")
        assert rc != 0

    def test_unblock_clears_reason(self, tmp_project, workspace):
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "blocked", "--reason", "broken")
        _run_tool(tmp_project, "unblock", "--slug", "test-exp")
        state_data, _ = _run_tool(tmp_project, "state", "--slug", "test-exp")
        assert state_data["workspace_state"]["blocked"] is False

    def test_unblock_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "unblock", "--slug", "nope")
        assert rc != 0

    def test_unblock_defaults_to_designed(self, tmp_project, workspace):
        _run_tool(tmp_project, "transition", "--slug", "test-exp", "--to", "blocked", "--reason", "early fail")
        data, rc = _run_tool(tmp_project, "unblock", "--slug", "test-exp")
        assert rc == 0
        assert data["reverted_to"] == "designed"


# ---------------------------------------------------------------------------
# State machine completeness
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_all_phases_defined(self):
        assert "designed" in ALL_PHASES
        assert "evaluated" in ALL_PHASES
        assert "blocked" in ALL_PHASES

    def test_blocked_can_return_to_any_phase(self):
        blocked_targets = VALID_TRANSITIONS["blocked"]
        assert "designed" in blocked_targets
        assert "validated" in blocked_targets

    def test_evaluated_is_terminal(self):
        assert VALID_TRANSITIONS["evaluated"] == []

    def test_baseline_checked_in_chain(self):
        assert "baseline_checked" in VALID_TRANSITIONS["designed"]
        assert "scaffolded" in VALID_TRANSITIONS["baseline_checked"]

    def test_scaffold_types_exist(self):
        assert "classification" in SCAFFOLD_TYPES
        assert "generation" in SCAFFOLD_TYPES
        assert "reinforcement" in SCAFFOLD_TYPES
        assert "llm-eval" in SCAFFOLD_TYPES
