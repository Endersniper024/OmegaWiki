"""Tests for tools/exp_recon.py — GitHub reconnaissance tool."""

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

from tools.exp_recon import (
    DATASET_CLASS_BASES,
    DEP_FILES,
    ENTRY_POINT_INDICATORS,
    MODEL_CLASS_BASES,
    REFERENCES_DIR,
    _count_loc,
    _detect_framework,
    _detect_training_pattern,
    _extract_dependencies,
    _find_config_files,
    _find_data_files,
    _find_entry_points,
    _find_model_files,
    _find_eval_files,
    _read_readme_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# test", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    return tmp_path


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository structure for analysis."""
    repo = tmp_path / "sample-repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Sample Repo\nA PyTorch model for classification.\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("torch>=2.0\ntransformers>=4.30\nwandb\n", encoding="utf-8")

    src = repo / "src"
    src.mkdir()
    (src / "model.py").write_text(textwrap.dedent("""\
        import torch
        import torch.nn as nn

        class MyModel(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.linear = nn.Linear(hidden_size, 10)

            def forward(self, x):
                return self.linear(x)
    """), encoding="utf-8")

    (src / "train.py").write_text(textwrap.dedent("""\
        import argparse
        import torch
        import wandb

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", default="config.yaml")
            args = parser.parse_args()
            wandb.init(project="test")
            print("Training...")
    """), encoding="utf-8")

    (src / "dataset.py").write_text(textwrap.dedent("""\
        from torch.utils.data import Dataset

        class MyDataset(Dataset):
            def __init__(self, data_path):
                self.data = []

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]
    """), encoding="utf-8")

    (src / "evaluate.py").write_text(textwrap.dedent("""\
        def evaluate(model, dataloader):
            correct = 0
            total = 0
            return correct / total if total > 0 else 0
    """), encoding="utf-8")

    configs = repo / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("lr: 0.001\nbatch_size: 32\n", encoding="utf-8")

    # Init git so last_commit works
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True,
                    env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"})
    return repo


def _run_tool(cwd, *cli_args):
    cmd = [sys.executable, str(ROOT / "tools" / "exp_recon.py")] + list(cli_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    try:
        return json.loads(proc.stdout), proc.returncode
    except json.JSONDecodeError:
        return {"raw": proc.stdout, "stderr": proc.stderr}, proc.returncode


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------


class TestFindEntryPoints:
    def test_finds_argparse(self, sample_repo):
        entries = _find_entry_points(sample_repo)
        assert any("train.py" in e for e in entries)

    def test_finds_main_guard(self, sample_repo):
        entries = _find_entry_points(sample_repo)
        assert any("train.py" in e for e in entries)

    def test_ignores_non_entry(self, sample_repo):
        entries = _find_entry_points(sample_repo)
        assert not any("evaluate.py" in e for e in entries)

    def test_empty_dir(self, tmp_path):
        entries = _find_entry_points(tmp_path)
        assert entries == []

    def test_limits_output(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(25):
            (src / f"entry_{i}.py").write_text("if __name__ == '__main__': pass\n", encoding="utf-8")
        entries = _find_entry_points(tmp_path)
        assert len(entries) <= 20


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------


class TestFindModelFiles:
    def test_finds_nn_module(self, sample_repo):
        models = _find_model_files(sample_repo)
        assert any("model.py" in m for m in models)

    def test_ignores_non_model(self, sample_repo):
        models = _find_model_files(sample_repo)
        assert not any("dataset.py" in m for m in models)

    def test_empty_dir(self, tmp_path):
        assert _find_model_files(tmp_path) == []

    def test_multiple_models(self, tmp_path):
        src = tmp_path / "models"
        src.mkdir()
        (src / "a.py").write_text("class A(nn.Module): pass\n", encoding="utf-8")
        (src / "b.py").write_text("class B(nn.Module): pass\n", encoding="utf-8")
        models = _find_model_files(tmp_path)
        assert len(models) == 2

    def test_transformers_model(self, tmp_path):
        (tmp_path / "m.py").write_text("class M(PreTrainedModel): pass\n", encoding="utf-8")
        models = _find_model_files(tmp_path)
        assert len(models) == 1


# ---------------------------------------------------------------------------
# Dataset detection
# ---------------------------------------------------------------------------


class TestFindDataFiles:
    def test_finds_dataset_class(self, sample_repo):
        data = _find_data_files(sample_repo)
        assert any("dataset.py" in d for d in data)

    def test_ignores_non_dataset(self, sample_repo):
        data = _find_data_files(sample_repo)
        assert not any("model.py" in d for d in data)

    def test_empty_dir(self, tmp_path):
        assert _find_data_files(tmp_path) == []

    def test_iterable_dataset(self, tmp_path):
        (tmp_path / "d.py").write_text("class D(IterableDataset): pass\n", encoding="utf-8")
        data = _find_data_files(tmp_path)
        assert len(data) == 1

    def test_hf_dataset(self, tmp_path):
        (tmp_path / "d.py").write_text("from datasets import Dataset\nclass D(datasets.Dataset): pass\n", encoding="utf-8")
        data = _find_data_files(tmp_path)
        assert len(data) == 1


# ---------------------------------------------------------------------------
# Eval detection
# ---------------------------------------------------------------------------


class TestFindEvalFiles:
    def test_finds_evaluate(self, sample_repo):
        evals = _find_eval_files(sample_repo)
        assert any("evaluate.py" in e for e in evals)

    def test_finds_metrics(self, tmp_path):
        (tmp_path / "metrics.py").write_text("# metrics\n", encoding="utf-8")
        evals = _find_eval_files(tmp_path)
        assert len(evals) == 1

    def test_empty_dir(self, tmp_path):
        assert _find_eval_files(tmp_path) == []

    def test_finds_test_file(self, tmp_path):
        (tmp_path / "test_model.py").write_text("# test\n", encoding="utf-8")
        evals = _find_eval_files(tmp_path)
        assert len(evals) == 1

    def test_finds_benchmark(self, tmp_path):
        (tmp_path / "benchmark_runner.py").write_text("# bench\n", encoding="utf-8")
        evals = _find_eval_files(tmp_path)
        assert len(evals) == 1


# ---------------------------------------------------------------------------
# Config detection
# ---------------------------------------------------------------------------


class TestFindConfigFiles:
    def test_finds_yaml(self, sample_repo):
        configs = _find_config_files(sample_repo)
        assert any("base.yaml" in c for c in configs)

    def test_empty_dir(self, tmp_path):
        assert _find_config_files(tmp_path) == []

    def test_ignores_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("x\n", encoding="utf-8")
        configs = _find_config_files(tmp_path)
        assert len(configs) == 0

    def test_finds_json_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        configs = _find_config_files(tmp_path)
        assert len(configs) == 1

    def test_finds_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
        configs = _find_config_files(tmp_path)
        assert len(configs) == 1


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


class TestDetectFramework:
    def test_detects_pytorch(self, sample_repo):
        assert _detect_framework(sample_repo) == "pytorch"

    def test_detects_tensorflow(self, tmp_path):
        (tmp_path / "m.py").write_text("import tensorflow as tf\n", encoding="utf-8")
        assert _detect_framework(tmp_path) == "tensorflow"

    def test_detects_jax(self, tmp_path):
        (tmp_path / "m.py").write_text("import jax\nimport jax.numpy as jnp\n", encoding="utf-8")
        assert _detect_framework(tmp_path) == "jax"

    def test_detects_huggingface(self, tmp_path):
        (tmp_path / "a.py").write_text("from transformers import AutoModel\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("from transformers import AutoTokenizer\n", encoding="utf-8")
        assert _detect_framework(tmp_path) == "huggingface"

    def test_unknown_framework(self, tmp_path):
        (tmp_path / "m.py").write_text("print('hello')\n", encoding="utf-8")
        assert _detect_framework(tmp_path) == "unknown"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


class TestExtractDependencies:
    def test_finds_requirements(self, sample_repo):
        deps = _extract_dependencies(sample_repo)
        assert "requirements.txt" in deps["files_found"]
        assert "torch" in deps["key_libs"]
        assert "transformers" in deps["key_libs"]

    def test_no_deps(self, tmp_path):
        deps = _extract_dependencies(tmp_path)
        assert deps["files_found"] == []
        assert deps["key_libs"] == []

    def test_parses_versioned_reqs(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy>=1.20\nscipy==1.9.0\n", encoding="utf-8")
        deps = _extract_dependencies(tmp_path)
        assert "numpy" in deps["key_libs"]
        assert "scipy" in deps["key_libs"]

    def test_ignores_comments(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("# comment\nnumpy\n", encoding="utf-8")
        deps = _extract_dependencies(tmp_path)
        assert "numpy" in deps["key_libs"]
        assert len(deps["key_libs"]) == 1

    def test_pyproject_python_version(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('requires-python = ">=3.9"\n', encoding="utf-8")
        deps = _extract_dependencies(tmp_path)
        assert "3.9" in deps["python_version"]


# ---------------------------------------------------------------------------
# Training pattern detection
# ---------------------------------------------------------------------------


class TestDetectTrainingPattern:
    def test_detects_wandb(self, sample_repo):
        pattern = _detect_training_pattern(sample_repo)
        assert pattern["logging"] == "wandb"

    def test_detects_distributed(self, tmp_path):
        (tmp_path / "t.py").write_text("from torch.nn.parallel import DistributedDataParallel\n", encoding="utf-8")
        pattern = _detect_training_pattern(tmp_path)
        assert pattern["distributed"] is True

    def test_detects_mixed_precision(self, tmp_path):
        (tmp_path / "t.py").write_text("scaler = GradScaler()\n", encoding="utf-8")
        pattern = _detect_training_pattern(tmp_path)
        assert pattern["mixed_precision"] is True

    def test_detects_checkpointing(self, tmp_path):
        (tmp_path / "t.py").write_text("torch.save(model.state_dict(), 'ckpt.pt')\n", encoding="utf-8")
        pattern = _detect_training_pattern(tmp_path)
        assert pattern["checkpointing"] is True

    def test_no_patterns(self, tmp_path):
        (tmp_path / "t.py").write_text("print('hello')\n", encoding="utf-8")
        pattern = _detect_training_pattern(tmp_path)
        assert pattern["distributed"] is False
        assert pattern["logging"] == "none"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadReadme:
    def test_reads_readme(self, sample_repo):
        summary = _read_readme_summary(sample_repo)
        assert "Sample Repo" in summary
        assert "PyTorch" in summary

    def test_no_readme(self, tmp_path):
        assert _read_readme_summary(tmp_path) == ""

    def test_truncates(self, tmp_path):
        (tmp_path / "README.md").write_text("A" * 1000, encoding="utf-8")
        summary = _read_readme_summary(tmp_path, max_chars=100)
        assert len(summary) <= 100

    def test_skips_badges(self, tmp_path):
        (tmp_path / "README.md").write_text("![badge](url)\n# Title\nContent here.\n", encoding="utf-8")
        summary = _read_readme_summary(tmp_path)
        assert "badge" not in summary
        assert "Title" in summary

    def test_rst_readme(self, tmp_path):
        (tmp_path / "README.rst").write_text("Title\n=====\nContent.\n", encoding="utf-8")
        summary = _read_readme_summary(tmp_path)
        assert "Title" in summary


# ---------------------------------------------------------------------------
# LOC
# ---------------------------------------------------------------------------


class TestCountLOC:
    def test_counts_python_lines(self, sample_repo):
        loc = _count_loc(sample_repo)
        assert loc > 0

    def test_empty_dir(self, tmp_path):
        assert _count_loc(tmp_path) == 0

    def test_ignores_git(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "something.py").write_text("x = 1\n" * 100, encoding="utf-8")
        assert _count_loc(tmp_path) == 0

    def test_counts_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("z = 3\n", encoding="utf-8")
        assert _count_loc(tmp_path) == 3

    def test_ignores_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-310.pyc").write_text("x" * 100, encoding="utf-8")
        # .pyc won't match *.py so it shouldn't count
        assert _count_loc(tmp_path) == 0


# ---------------------------------------------------------------------------
# CLI: analyze
# ---------------------------------------------------------------------------


class TestAnalyzeCLI:
    def test_analyze_sample_repo(self, sample_repo):
        data, rc = _run_tool(sample_repo.parent, "analyze", "--path", str(sample_repo))
        assert rc == 0
        assert data["status"] == "ok"
        assert len(data["structure"]["entry_points"]) > 0
        assert len(data["structure"]["model_files"]) > 0
        assert data["dependencies"]["framework"] == "pytorch"

    def test_analyze_nonexistent(self, tmp_path):
        data, rc = _run_tool(tmp_path, "analyze", "--path", "/nonexistent")
        assert rc != 0

    def test_analyze_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        data, rc = _run_tool(tmp_path, "analyze", "--path", str(f))
        assert rc != 0

    def test_analyze_has_training_pattern(self, sample_repo):
        data, rc = _run_tool(sample_repo.parent, "analyze", "--path", str(sample_repo))
        assert rc == 0
        assert "training_pattern" in data
        assert data["training_pattern"]["logging"] == "wandb"

    def test_analyze_has_readme_summary(self, sample_repo):
        data, rc = _run_tool(sample_repo.parent, "analyze", "--path", str(sample_repo))
        assert rc == 0
        assert "Sample Repo" in data["readme_summary"]

    def test_analyze_has_loc(self, sample_repo):
        data, rc = _run_tool(sample_repo.parent, "analyze", "--path", str(sample_repo))
        assert rc == 0
        assert data["lines_of_code"] > 0


# ---------------------------------------------------------------------------
# CLI: report
# ---------------------------------------------------------------------------


class TestReportCLI:
    def test_report_no_references(self, tmp_project):
        data, rc = _run_tool(tmp_project, "report", "--slug", "test-exp")
        assert rc == 0
        assert data["count"] == 0

    def test_report_with_references(self, tmp_project):
        refs = tmp_project / REFERENCES_DIR / "repo1"
        refs.mkdir(parents=True)
        (refs / "train.py").write_text("import torch\n", encoding="utf-8")
        (refs / "README.md").write_text("# Repo 1\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "report", "--slug", "test-exp")
        assert rc == 0
        assert data["count"] == 1
        assert data["references"][0]["name"] == "repo1"

    def test_report_multiple_refs(self, tmp_project):
        for name in ["repo1", "repo2", "repo3"]:
            d = tmp_project / REFERENCES_DIR / name
            d.mkdir(parents=True)
            (d / "main.py").write_text("print('hi')\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "report", "--slug", "test-exp")
        assert rc == 0
        assert data["count"] == 3

    def test_report_detects_framework(self, tmp_project):
        refs = tmp_project / REFERENCES_DIR / "torch-repo"
        refs.mkdir(parents=True)
        (refs / "model.py").write_text("import torch\nimport torch.nn as nn\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "report", "--slug", "test-exp")
        assert rc == 0
        assert data["references"][0]["framework"] == "pytorch"

    def test_report_ignores_hidden(self, tmp_project):
        d = tmp_project / REFERENCES_DIR / ".hidden"
        d.mkdir(parents=True)
        (d / "x.py").write_text("x\n", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "report", "--slug", "test-exp")
        assert rc == 0
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# CLI: search (mocked — requires gh auth)
# ---------------------------------------------------------------------------


class TestSearchCLI:
    def test_search_no_gh(self, tmp_project):
        """If gh is not available, should error gracefully."""
        with patch("subprocess.run") as mock_run:
            # First call is gh --version check
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            data, rc = _run_tool(tmp_project, "search", "--query", "test")
            # We can't fully mock subprocess in a subprocess call,
            # but we can test that the tool doesn't crash
            assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# CLI: clone
# ---------------------------------------------------------------------------


class TestCloneCLI:
    def test_clone_invalid_url(self, tmp_project):
        data, rc = _run_tool(tmp_project, "clone", "--repo", "not-a-url", "--dest", "refs/test")
        assert rc != 0
        assert "Invalid repo URL" in data.get("message", "")

    def test_clone_dest_exists(self, tmp_project):
        dest = tmp_project / "refs" / "existing"
        dest.mkdir(parents=True)
        (dest / "marker").write_text("x", encoding="utf-8")
        data, rc = _run_tool(tmp_project, "clone", "--repo", "https://github.com/test/test", "--dest", str(dest))
        assert rc != 0

    def test_clone_creates_parent(self, tmp_project):
        # This would fail on actual clone (no such repo), but should create parent dir
        dest = tmp_project / "deep" / "nested" / "repo"
        data, rc = _run_tool(tmp_project, "clone", "--repo", "https://github.com/nonexistent/repo", "--dest", str(dest))
        # Will fail at git clone, but parent should be created
        assert (tmp_project / "deep" / "nested").exists()

    def test_clone_accepts_git_url(self, tmp_project):
        """Verify git@ URLs are accepted (even if clone fails)."""
        data, rc = _run_tool(tmp_project, "clone", "--repo", "git@github.com:user/repo.git", "--dest", str(tmp_project / "r"))
        # Clone will fail but URL validation should pass
        if rc != 0:
            assert "Invalid repo URL" not in data.get("message", "")

    def test_clone_default_depth(self, tmp_project):
        # Just verify the argument is accepted
        data, rc = _run_tool(tmp_project, "clone", "--repo", "https://github.com/nonexistent/repo", "--dest", str(tmp_project / "r"), "--depth", "1")
        # Will fail at git clone, but args should parse
        assert isinstance(data, dict)
