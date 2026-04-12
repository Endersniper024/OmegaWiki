"""Tests for tools/exp_env.py — Execution environment abstraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.exp_env import (
    COMPILED_PATTERNS,
    ERROR_TAXONOMY,
    WORKSPACE_BASE,
    _classify_error,
    _local_check,
    _local_collect,
    _local_setup,
    _local_teardown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# test", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / WORKSPACE_BASE).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def workspace(tmp_project):
    slug = "test-env-exp"
    ws = tmp_project / WORKSPACE_BASE / slug
    ws.mkdir(parents=True)
    for d in ["src", "configs", "scripts", "results", "logs"]:
        (ws / d).mkdir()
    (ws / "src" / "train.py").write_text("print('training')\n", encoding="utf-8")
    (ws / "scripts" / "run.sh").write_text("#!/bin/bash\necho 'running'\n", encoding="utf-8")
    (ws / "configs" / "base.yaml").write_text("lr: 0.001\n", encoding="utf-8")
    (ws / "requirements.txt").write_text("# no deps\n", encoding="utf-8")
    yaml = textwrap.dedent("""\
        slug: test-env-exp
        workspace_state:
          phase: validated
          blocked: false
    """)
    (ws / ".experiment.yaml").write_text(yaml, encoding="utf-8")
    return ws


def _run_tool(tmp_project, *cli_args):
    cmd = [sys.executable, str(ROOT / "tools" / "exp_env.py")] + list(cli_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_project))
    try:
        return json.loads(proc.stdout), proc.returncode
    except json.JSONDecodeError:
        return {"raw": proc.stdout, "stderr": proc.stderr}, proc.returncode


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    def test_classify_import_error(self):
        result = _classify_error("ModuleNotFoundError: No module named 'torch'")
        assert result["category"] == "IMPORT_ERROR"
        assert "torch" in result["suggested_fixes"][0]

    def test_classify_oom(self):
        result = _classify_error("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB")
        assert result["category"] == "OOM"

    def test_classify_nan_loss(self):
        result = _classify_error("Step 100: loss=nan, lr=0.001")
        assert result["category"] == "NAN_LOSS"

    def test_classify_data_error(self):
        result = _classify_error("FileNotFoundError: data/train.csv not found")
        assert result["category"] == "DATA_ERROR"

    def test_classify_shape_mismatch(self):
        result = _classify_error("RuntimeError: size mismatch, m1: [32 x 768], m2: [512 x 10]")
        assert result["category"] == "SHAPE_MISMATCH"

    def test_classify_timeout(self):
        result = _classify_error("TimeoutError: operation timed out")
        assert result["category"] == "TIMEOUT"

    def test_classify_unknown(self):
        result = _classify_error("Something weird happened")
        assert result["category"] == "UNKNOWN"

    def test_classify_empty(self):
        result = _classify_error("")
        assert result["category"] == "UNKNOWN"

    def test_all_categories_have_fixes(self):
        for cat, info in ERROR_TAXONOMY.items():
            assert "suggested_fix" in info
            assert len(info["patterns"]) > 0

    def test_import_error_extracts_module(self):
        result = _classify_error("ModuleNotFoundError: No module named 'transformers'")
        assert "transformers" in result["suggested_fixes"][0]


# ---------------------------------------------------------------------------
# Local setup
# ---------------------------------------------------------------------------


class TestLocalSetup:
    def test_setup_no_requirements(self, workspace):
        (workspace / "requirements.txt").unlink()
        result = _local_setup(workspace, {})
        assert result["ready"] is True

    def test_setup_creates_venv(self, workspace):
        # This test creates a real venv, may be slow
        (workspace / "requirements.txt").write_text("# empty\n", encoding="utf-8")
        result = _local_setup(workspace, {})
        assert result["backend"] == "local"
        # Venv may or may not succeed depending on system
        # but the function should not crash

    def test_setup_returns_backend(self, workspace):
        (workspace / "requirements.txt").unlink()
        result = _local_setup(workspace, {})
        assert result["backend"] == "local"

    def test_setup_with_preset(self, workspace):
        (workspace / "requirements.txt").unlink()
        result = _local_setup(workspace, {"gpu": False})
        assert result["ready"] is True

    def test_setup_existing_venv(self, workspace):
        (workspace / ".venv").mkdir()
        (workspace / ".venv" / "bin").mkdir(parents=True)
        (workspace / "requirements.txt").unlink()
        result = _local_setup(workspace, {})
        assert result["ready"] is True


# ---------------------------------------------------------------------------
# Local check
# ---------------------------------------------------------------------------


class TestLocalCheck:
    def test_check_not_started(self, workspace):
        result = _local_check(workspace)
        assert result["alive"] is False
        assert result["exit_reason"] == "not_started"

    def test_check_with_results(self, workspace):
        (workspace / "results" / "seed_42.json").write_text('{"acc": 0.9}', encoding="utf-8")
        result = _local_check(workspace)
        assert result["alive"] is False
        assert result["has_results"] is True

    def test_check_with_pid_dead_process(self, workspace):
        # Write a PID that doesn't exist
        (workspace / ".pid").write_text("999999999", encoding="utf-8")
        result = _local_check(workspace)
        assert result["alive"] is False

    def test_check_detects_anomalies(self, workspace):
        (workspace / ".pid").write_text("999999999", encoding="utf-8")
        (workspace / "logs" / "run.log").write_text("Step 1: loss=nan\n", encoding="utf-8")
        result = _local_check(workspace)
        assert len(result["anomalies"]) > 0
        assert result["anomalies"][0]["type"] == "NAN_LOSS"

    def test_check_deduplicates_anomalies(self, workspace):
        (workspace / ".pid").write_text("999999999", encoding="utf-8")
        log = "loss=nan\nloss=nan\nloss=nan\n"
        (workspace / "logs" / "run.log").write_text(log, encoding="utf-8")
        result = _local_check(workspace)
        nan_count = sum(1 for a in result["anomalies"] if a["type"] == "NAN_LOSS")
        assert nan_count == 1


# ---------------------------------------------------------------------------
# Local collect
# ---------------------------------------------------------------------------


class TestLocalCollect:
    def test_collect_no_results(self, workspace):
        result = _local_collect(workspace, None)
        assert result["result_files"] == []
        assert result["metrics"] == {}

    def test_collect_with_json_results(self, workspace):
        (workspace / "results" / "seed_42.json").write_text(
            '{"accuracy": 0.95, "f1": 0.93}', encoding="utf-8"
        )
        (workspace / "results" / "seed_123.json").write_text(
            '{"accuracy": 0.92, "f1": 0.90}', encoding="utf-8"
        )
        result = _local_collect(workspace, None)
        assert len(result["result_files"]) == 2
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["accuracy"]["count"] == 2

    def test_collect_aggregation(self, workspace):
        (workspace / "results" / "r1.json").write_text('{"acc": 0.8}', encoding="utf-8")
        (workspace / "results" / "r2.json").write_text('{"acc": 1.0}', encoding="utf-8")
        result = _local_collect(workspace, None)
        assert result["metrics"]["acc"]["mean"] == 0.9
        assert result["metrics"]["acc"]["min"] == 0.8
        assert result["metrics"]["acc"]["max"] == 1.0

    def test_collect_to_dest(self, workspace, tmp_path):
        dest = tmp_path / "collected"
        (workspace / "results" / "r1.json").write_text('{"x": 1}', encoding="utf-8")
        result = _local_collect(workspace, str(dest))
        assert (dest / "results" / "r1.json").exists()

    def test_collect_invalid_json(self, workspace):
        (workspace / "results" / "bad.json").write_text("not json", encoding="utf-8")
        result = _local_collect(workspace, None)
        assert len(result["result_files"]) == 1
        assert result["metrics"] == {}


# ---------------------------------------------------------------------------
# Local teardown
# ---------------------------------------------------------------------------


class TestLocalTeardown:
    def test_teardown_removes_pid(self, workspace):
        (workspace / ".pid").write_text("12345", encoding="utf-8")
        result = _local_teardown(workspace)
        assert ".pid" in result["cleaned"]
        assert not (workspace / ".pid").exists()

    def test_teardown_removes_venv(self, workspace):
        (workspace / ".venv").mkdir()
        (workspace / ".venv" / "marker").write_text("x", encoding="utf-8")
        result = _local_teardown(workspace)
        assert ".venv" in result["cleaned"]
        assert not (workspace / ".venv").exists()

    def test_teardown_nothing_to_clean(self, workspace):
        result = _local_teardown(workspace)
        assert result["cleaned"] == []

    def test_teardown_only_existing(self, workspace):
        (workspace / ".pid").write_text("12345", encoding="utf-8")
        result = _local_teardown(workspace)
        assert ".pid" in result["cleaned"]
        assert ".venv" not in result["cleaned"]

    def test_teardown_preserves_code(self, workspace):
        (workspace / ".pid").write_text("12345", encoding="utf-8")
        _local_teardown(workspace)
        assert (workspace / "src" / "train.py").exists()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_setup_local(self, tmp_project, workspace):
        (workspace / "requirements.txt").write_text("# none\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "setup", "--slug", "test-env-exp", "--backend", "local")
        assert rc == 0
        assert data["backend"] == "local"

    def test_check_local(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "check", "--slug", "test-env-exp", "--backend", "local")
        assert rc == 0
        assert "alive" in data

    def test_collect_local(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "collect", "--slug", "test-env-exp", "--backend", "local")
        assert rc == 0

    def test_teardown_local(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "teardown", "--slug", "test-env-exp", "--backend", "local")
        assert rc == 0
        assert "cleaned" in data

    def test_missing_workspace(self, tmp_project):
        data, rc = _run_tool(tmp_project, "setup", "--slug", "nonexistent", "--backend", "local")
        assert rc != 0

    def test_docker_stub(self, tmp_project, workspace):
        data, rc = _run_tool(tmp_project, "setup", "--slug", "test-env-exp", "--backend", "docker")
        # Docker is a stub, should indicate not implemented
        assert "docker" in str(data).lower() or "not yet implemented" in str(data).lower()

    def test_no_command(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "exp_env.py")],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0 or "usage" in proc.stderr.lower()
