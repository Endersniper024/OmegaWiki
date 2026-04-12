#!/usr/bin/env python3
"""GitHub Reconnaissance Tool.

Structured search, clone, and analysis of GitHub repositories for
experiment reference and baseline discovery.  All output is structured
JSON following the ACI principle.

Called by skills via:  Bash: python3 tools/exp_recon.py <command> [args]

Commands:
    search   --query Q [--max N]             Search GitHub repos (via gh CLI)
    clone    --repo URL --dest D [--depth N] Clone repo to local references/
    analyze  --path P                        Static analysis of a cloned repo
    report   --slug S                        Aggregate analysis of all references for an experiment
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCES_DIR = "references"

# File patterns for static analysis
ENTRY_POINT_INDICATORS = [
    "if __name__",
    "argparse.ArgumentParser",
    "click.command",
    "typer.run",
    "fire.Fire",
]

MODEL_CLASS_BASES = [
    "nn.Module", "torch.nn.Module",
    "PreTrainedModel", "transformers.PreTrainedModel",
    "LightningModule", "pl.LightningModule",
    "keras.Model", "tf.keras.Model",
]

DATASET_CLASS_BASES = [
    "Dataset", "torch.utils.data.Dataset",
    "IterableDataset",
    "datasets.Dataset",
]

CONFIG_PATTERNS = ["*.yaml", "*.yml", "*.json", "*.toml", "*.cfg", "*.ini"]

DEP_FILES = [
    "requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
    "setup.py", "setup.cfg", "pyproject.toml",
    "Pipfile", "conda.yaml", "environment.yml", "environment.yaml",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "CLAUDE.md").exists() and (p / "tools").is_dir():
            return p
    return cwd


def _ok(data: dict) -> None:
    data["status"] = "ok"
    print(json.dumps(data, ensure_ascii=False, indent=2))
    sys.exit(0)


def _error(msg: str, **extra) -> None:
    out = {"status": "error", "message": msg}
    out.update(extra)
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace) -> None:
    """Search GitHub repositories via gh CLI."""
    query = args.query
    max_results = args.max or 10

    # Check if gh is available
    gh_check = subprocess.run(["gh", "--version"], capture_output=True, text=True)
    if gh_check.returncode != 0:
        _error("GitHub CLI (gh) not found. Install: https://cli.github.com/",
               suggested_fix="Install gh CLI and run 'gh auth login'")

    # Search via gh
    try:
        proc = subprocess.run(
            [
                "gh", "search", "repos", query,
                "--limit", str(max_results),
                "--json", "fullName,description,stargazersCount,updatedAt,licenseInfo,repositoryTopics,url",
            ],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _error("GitHub search timed out after 30s")
    except FileNotFoundError:
        _error("gh command not found")

    if proc.returncode != 0:
        _error(f"gh search failed: {proc.stderr.strip()}",
               suggested_fix="Run 'gh auth login' to authenticate")

    try:
        repos = json.loads(proc.stdout)
    except json.JSONDecodeError:
        _error(f"Failed to parse gh output: {proc.stdout[:200]}")

    # Normalize output
    results = []
    for r in repos:
        license_info = r.get("licenseInfo") or {}
        topics = r.get("repositoryTopics") or []
        if isinstance(topics, list) and topics and isinstance(topics[0], dict):
            topics = [t.get("name", "") for t in topics]

        results.append({
            "repo": r.get("fullName", ""),
            "url": r.get("url", ""),
            "description": (r.get("description") or "")[:200],
            "stars": r.get("stargazersCount", 0),
            "updated": r.get("updatedAt", ""),
            "license": license_info.get("key", "unknown") if isinstance(license_info, dict) else str(license_info),
            "topics": topics,
        })

    _ok({
        "query": query,
        "count": len(results),
        "repos": results,
    })


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------


def cmd_clone(args: argparse.Namespace) -> None:
    """Clone a repository to local references/ directory."""
    repo_url = args.repo
    dest = args.dest
    depth = args.depth or 1

    # Validate URL
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        _error(f"Invalid repo URL: {repo_url}",
               suggested_fix="Use https://github.com/user/repo or git@github.com:user/repo.git")

    dest_path = Path(dest)
    if dest_path.exists() and any(dest_path.iterdir()):
        _error(f"Destination already exists: {dest}",
               suggested_fix="Remove existing directory or use a different --dest")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Clone
    cmd = ["git", "clone", "--depth", str(depth)]
    cmd += [repo_url, str(dest_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        _error("git clone timed out after 120s")

    if proc.returncode != 0:
        _error(f"git clone failed: {proc.stderr.strip()}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in dest_path.rglob("*") if f.is_file())
    size_mb = round(total_size / (1024 * 1024), 1)

    _ok({
        "repo": repo_url,
        "path": str(dest_path),
        "depth": depth,
        "size_mb": size_mb,
    })


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


def _find_entry_points(root: Path) -> list[str]:
    """Find Python files that look like entry points."""
    entries = []
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            if any(ind in content for ind in ENTRY_POINT_INDICATORS):
                entries.append(str(py.relative_to(root)))
        except (OSError, UnicodeDecodeError):
            pass
    return entries[:20]  # Limit output


def _find_model_files(root: Path) -> list[str]:
    """Find files containing model class definitions."""
    models = []
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            if any(base in content for base in MODEL_CLASS_BASES):
                models.append(str(py.relative_to(root)))
        except (OSError, UnicodeDecodeError):
            pass
    return models[:20]


def _find_data_files(root: Path) -> list[str]:
    """Find files containing dataset class definitions."""
    data_files = []
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            if any(base in content for base in DATASET_CLASS_BASES):
                data_files.append(str(py.relative_to(root)))
        except (OSError, UnicodeDecodeError):
            pass
    return data_files[:20]


def _find_eval_files(root: Path) -> list[str]:
    """Find evaluation-related files."""
    eval_files = []
    eval_names = {"eval", "evaluate", "evaluation", "metrics", "test", "benchmark"}
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        stem = py.stem.lower()
        if any(e in stem for e in eval_names):
            eval_files.append(str(py.relative_to(root)))
    return eval_files[:20]


def _find_config_files(root: Path) -> list[str]:
    """Find configuration files."""
    configs = []
    for pattern in CONFIG_PATTERNS:
        for f in root.rglob(pattern):
            if ".git" in f.parts or "__pycache__" in f.parts or "node_modules" in f.parts:
                continue
            configs.append(str(f.relative_to(root)))
    return configs[:30]


def _detect_framework(root: Path) -> str:
    """Detect ML framework used."""
    indicators = {
        "pytorch": ["import torch", "from torch", "nn.Module"],
        "tensorflow": ["import tensorflow", "from tensorflow", "tf.keras"],
        "jax": ["import jax", "from jax", "flax.linen"],
        "huggingface": ["from transformers", "import transformers"],
    }
    counts = {k: 0 for k in indicators}
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            for framework, patterns in indicators.items():
                if any(p in content for p in patterns):
                    counts[framework] += 1
        except (OSError, UnicodeDecodeError):
            pass
    if not any(counts.values()):
        return "unknown"
    return max(counts, key=counts.get)


def _extract_dependencies(root: Path) -> dict:
    """Extract dependency information."""
    deps = {"files_found": [], "key_libs": [], "python_version": ""}

    for dep_file in DEP_FILES:
        path = root / dep_file
        if path.exists():
            deps["files_found"].append(dep_file)

    # Parse requirements.txt
    req = root / "requirements.txt"
    if req.exists():
        try:
            lines = req.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    pkg = line.split(">=")[0].split("==")[0].split("<=")[0].split("<")[0].split(">")[0].split("[")[0].strip()
                    if pkg:
                        deps["key_libs"].append(pkg)
        except OSError:
            pass

    # Parse pyproject.toml for python version
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            for line in content.splitlines():
                if "requires-python" in line.lower():
                    deps["python_version"] = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    break
        except OSError:
            pass

    return deps


def _read_readme_summary(root: Path, max_chars: int = 500) -> str:
    """Read first max_chars of README."""
    for name in ["README.md", "README.rst", "README.txt", "README"]:
        readme = root / name
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="ignore")
                # Skip badges and header
                lines = text.splitlines()
                content_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("![") and not stripped.startswith("[!["):
                        content_lines.append(stripped)
                return " ".join(content_lines)[:max_chars]
            except OSError:
                pass
    return ""


def _count_loc(root: Path) -> int:
    """Count lines of Python code."""
    total = 0
    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts or "node_modules" in py.parts:
            continue
        try:
            total += sum(1 for _ in py.open(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    return total


def _detect_training_pattern(root: Path) -> dict:
    """Detect training patterns (distributed, mixed precision, etc.)."""
    pattern = {
        "distributed": False,
        "mixed_precision": False,
        "checkpointing": False,
        "logging": "none",
    }

    dist_indicators = ["DistributedDataParallel", "torch.distributed", "deepspeed", "accelerate"]
    amp_indicators = ["amp.autocast", "GradScaler", "mixed_precision", "fp16"]
    ckpt_indicators = ["save_checkpoint", "torch.save", "save_pretrained"]
    log_indicators = {
        "wandb": ["import wandb", "wandb.init", "wandb.log"],
        "tensorboard": ["SummaryWriter", "tensorboard"],
        "mlflow": ["import mlflow", "mlflow.log"],
    }

    for py in root.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            if any(ind in content for ind in dist_indicators):
                pattern["distributed"] = True
            if any(ind in content for ind in amp_indicators):
                pattern["mixed_precision"] = True
            if any(ind in content for ind in ckpt_indicators):
                pattern["checkpointing"] = True
            for logger, indicators in log_indicators.items():
                if any(ind in content for ind in indicators):
                    pattern["logging"] = logger
        except (OSError, UnicodeDecodeError):
            pass

    return pattern


def _get_last_commit(root: Path) -> str:
    """Get last commit date."""
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()[:10]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def cmd_analyze(args: argparse.Namespace) -> None:
    """Static analysis of a cloned repository."""
    root = Path(args.path).resolve()

    if not root.exists():
        _error(f"Path not found: {args.path}")
    if not root.is_dir():
        _error(f"Not a directory: {args.path}")

    structure = {
        "entry_points": _find_entry_points(root),
        "model_files": _find_model_files(root),
        "data_files": _find_data_files(root),
        "eval_files": _find_eval_files(root),
        "config_files": _find_config_files(root),
        "config_format": "",
    }

    # Determine primary config format
    for ext in ["yaml", "yml", "json", "toml"]:
        if any(f.endswith(f".{ext}") for f in structure["config_files"]):
            structure["config_format"] = ext if ext != "yml" else "yaml"
            break

    deps = _extract_dependencies(root)
    framework = _detect_framework(root)
    training = _detect_training_pattern(root)
    readme = _read_readme_summary(root)
    loc = _count_loc(root)
    last_commit = _get_last_commit(root)

    _ok({
        "path": str(root),
        "structure": structure,
        "dependencies": {
            "framework": framework,
            "key_libs": deps["key_libs"][:30],
            "python_version": deps["python_version"],
            "dep_files": deps["files_found"],
        },
        "training_pattern": training,
        "readme_summary": readme,
        "lines_of_code": loc,
        "last_commit": last_commit,
    })


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    """Aggregate analysis of all references for an experiment."""
    slug = args.slug
    root = _find_project_root()
    refs_dir = root / REFERENCES_DIR

    if not refs_dir.exists():
        _ok({"slug": slug, "references": [], "count": 0})

    # Also check experiment-specific baselines
    ws_baselines = root / "experiments" / "code" / slug / "baselines"

    analyses = []
    for d in sorted(refs_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            analysis = {
                "name": d.name,
                "path": str(d),
                "type": "reference",
            }
            # Quick stats
            py_count = sum(1 for _ in d.rglob("*.py") if ".git" not in _.parts)
            analysis["python_files"] = py_count
            analysis["framework"] = _detect_framework(d)
            analysis["readme_summary"] = _read_readme_summary(d, max_chars=200)
            analyses.append(analysis)

    if ws_baselines.exists():
        for d in sorted(ws_baselines.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                analysis = {
                    "name": d.name,
                    "path": str(d),
                    "type": "baseline",
                    "python_files": sum(1 for _ in d.rglob("*.py") if ".git" not in _.parts),
                    "framework": _detect_framework(d),
                    "readme_summary": _read_readme_summary(d, max_chars=200),
                }
                analyses.append(analysis)

    _ok({
        "slug": slug,
        "count": len(analyses),
        "references": analyses,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GitHub reconnaissance tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # search
    p = sub.add_parser("search", help="Search GitHub repos")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--max", type=int, default=10, help="Max results (default: 10)")

    # clone
    p = sub.add_parser("clone", help="Clone a repository")
    p.add_argument("--repo", required=True, help="Repository URL")
    p.add_argument("--dest", required=True, help="Local destination path")
    p.add_argument("--depth", type=int, default=1, help="Clone depth (default: 1, shallow)")

    # analyze
    p = sub.add_parser("analyze", help="Analyze a cloned repo")
    p.add_argument("--path", required=True, help="Path to cloned repo")

    # report
    p = sub.add_parser("report", help="Aggregate report for experiment references")
    p.add_argument("--slug", required=True, help="Experiment slug")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "search": cmd_search,
        "clone": cmd_clone,
        "analyze": cmd_analyze,
        "report": cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
