#!/usr/bin/env python3
"""Experiment Workspace Manager.

Manages experiment workspaces under experiments/code/{slug}/:
standard directory scaffolding, git versioning (snapshot/diff),
configuration composition, structural validation, and a workspace
state machine that tracks experiment lifecycle phases.

Called by skills via:  Bash: python3 tools/exp_workspace.py <command> [args]

Commands:
    scaffold       --slug S [--type T]          Create standard workspace directory
    validate       --slug S                     Check workspace structural completeness
    snapshot       --slug S --tag T [--message M] Git commit + tag
    diff           --slug S                     Show changes since last snapshot
    config-compose --slug S --base B [--override O] Merge YAML configs
    status         --slug S                     Workspace health summary
    state          --slug S                     Read workspace_state from .experiment.yaml
    transition     --slug S --to PHASE [--reason R] State machine transition
    unblock        --slug S                     Clear blocked state, revert to previous phase
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_BASE = "experiments/code"

REQUIRED_DIRS = ["src", "configs", "scripts", "results", "logs"]

REQUIRED_FILES = {
    "src/train.py": "training entry point",
    "configs/base.yaml": "base configuration",
    "scripts/run.sh": "run script",
    "requirements.txt": "Python dependencies",
}

OPTIONAL_FILES = {
    "src/model.py": "model definition",
    "src/dataset.py": "data loading",
    "src/evaluate.py": "evaluation logic",
    "src/utils.py": "utilities",
    "configs/pilot.yaml": "pilot configuration",
    "scripts/run_pilot.sh": "pilot run script",
    "scripts/setup_env.sh": "environment setup script",
    "README.md": "workspace documentation",
}

EXPERIMENT_YAML_TEMPLATE = """\
slug: {slug}
claim: ""
idea: ""
type: validation
stage: 1

environment: local-cpu
template: ""
overrides: {{}}

tracking: wandb
tracking_project: ""

seeds: [42, 123, 456]
pilot:
  max_epochs: 2
  max_samples: 1000
  timeout_minutes: 15

references: []
baselines: []

workspace_state:
  phase: {initial_phase}
  phases_completed: {{}}
  fix_rounds_total: 0
  blocked: false
  blocked_reason: ""
  recon: {{}}
  baseline_results: {{}}
  scaffold_strategy: ""
  deploy:
    backend: ""
    server: ""
    gpu: ""
    session: ""
    started: ""
    config: ""
"""

# Valid phase transitions: {from_phase: [allowed_to_phases]}
VALID_TRANSITIONS = {
    "designed": ["baseline_checked", "scaffolded", "blocked"],
    "baseline_checked": ["scaffolded", "blocked"],
    "scaffolded": ["validated", "blocked"],
    "validated": ["running", "scaffolded", "blocked"],
    "running": ["completed", "blocked"],
    "completed": ["collected", "blocked"],
    "collected": ["evaluated", "blocked"],
    "evaluated": [],
    "blocked": [
        "designed", "baseline_checked", "scaffolded", "validated",
        "running", "completed", "collected",
    ],
}

ALL_PHASES = list(VALID_TRANSITIONS.keys())

SCAFFOLD_TYPES = {
    "classification": {
        "src/model.py": "model definition (classifier)",
        "src/dataset.py": "dataset loader",
        "src/evaluate.py": "evaluation (accuracy, F1)",
    },
    "generation": {
        "src/model.py": "model definition (generator)",
        "src/dataset.py": "dataset loader",
        "src/evaluate.py": "evaluation (BLEU, ROUGE)",
    },
    "reinforcement": {
        "src/agent.py": "RL agent",
        "src/environment.py": "environment wrapper",
        "src/evaluate.py": "evaluation (reward, episode length)",
    },
    "llm-eval": {
        "src/evaluate.py": "LLM evaluation harness",
        "src/prompts.py": "prompt templates",
        "src/dataset.py": "benchmark loader",
    },
    "default": {},
}

# ---------------------------------------------------------------------------
# Lightweight YAML helpers (no PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_yaml_simple(text: str) -> dict:
    """Parse simple YAML into dict. Handles scalars, inline lists/dicts, block maps."""
    from remote import _parse_yaml  # reuse remote.py's parser
    return _parse_yaml(text)


def _dump_yaml_simple(data: dict, indent: int = 0) -> str:
    """Dump dict to simple YAML string."""
    lines = []
    prefix = "  " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            if not v:
                lines.append(f"{prefix}{k}: {{}}")
            else:
                lines.append(f"{prefix}{k}:")
                lines.append(_dump_yaml_simple(v, indent + 1))
        elif isinstance(v, list):
            if not v:
                lines.append(f"{prefix}{k}: []")
            elif all(isinstance(i, (str, int, float, bool)) for i in v):
                items = ", ".join(_format_scalar(i) for i in v)
                lines.append(f"{prefix}{k}: [{items}]")
            else:
                lines.append(f"{prefix}{k}:")
                for item in v:
                    if isinstance(item, dict):
                        first = True
                        for ik, iv in item.items():
                            if first:
                                lines.append(f"{prefix}  - {ik}: {_format_scalar(iv)}")
                                first = False
                            else:
                                lines.append(f"{prefix}    {ik}: {_format_scalar(iv)}")
                    else:
                        lines.append(f"{prefix}  - {_format_scalar(item)}")
        else:
            lines.append(f"{prefix}{k}: {_format_scalar(v)}")
    return "\n".join(lines)


def _format_scalar(v) -> str:
    if v is None or v == "":
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#{}[],'\"") or s in ("true", "false", "null"):
        return f'"{s}"'
    return s


def _read_experiment_yaml(ws_path: Path) -> dict:
    """Read .experiment.yaml from workspace."""
    yaml_path = ws_path / ".experiment.yaml"
    if not yaml_path.exists():
        return {}
    return _parse_yaml_simple(yaml_path.read_text(encoding="utf-8"))


def _write_experiment_yaml(ws_path: Path, data: dict) -> None:
    """Write .experiment.yaml to workspace."""
    yaml_path = ws_path / ".experiment.yaml"
    yaml_path.write_text(_dump_yaml_simple(data) + "\n", encoding="utf-8")


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Workspace path helpers
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Find OmegaWiki project root (directory containing CLAUDE.md)."""
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "CLAUDE.md").exists() and (p / "tools").is_dir():
            return p
    return cwd


def _workspace_path(slug: str) -> Path:
    """Get workspace path for a given experiment slug."""
    root = _find_project_root()
    return root / WORKSPACE_BASE / slug


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


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
# Commands
# ---------------------------------------------------------------------------


def cmd_scaffold(args: argparse.Namespace) -> None:
    """Create standard workspace directory structure."""
    slug = args.slug
    ws = _workspace_path(slug)
    exp_type = args.type or "default"

    if ws.exists() and any(ws.iterdir()):
        _error(f"Workspace already exists: {ws}", suggested_fix="Use a different slug or remove existing workspace")

    ws.mkdir(parents=True, exist_ok=True)

    # Create required directories
    created_dirs = []
    for d in REQUIRED_DIRS:
        (ws / d).mkdir(parents=True, exist_ok=True)
        created_dirs.append(d)

    # Create trajectory directory
    traj = ws / ".trajectory"
    traj.mkdir(exist_ok=True)
    (traj / "events.jsonl").touch()
    (traj / "fixes.jsonl").touch()

    # Create placeholder files
    created_files = []
    for f, desc in REQUIRED_FILES.items():
        fp = ws / f
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(f"# {desc}\n# TODO: implement\n", encoding="utf-8")
            created_files.append(f)

    # Type-specific files
    type_files = SCAFFOLD_TYPES.get(exp_type, SCAFFOLD_TYPES["default"])
    for f, desc in type_files.items():
        fp = ws / f
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(f"# {desc}\n# TODO: implement\n", encoding="utf-8")
            created_files.append(f)

    # Write .experiment.yaml
    yaml_content = EXPERIMENT_YAML_TEMPLATE.format(slug=slug, initial_phase="scaffolded")
    (ws / ".experiment.yaml").write_text(yaml_content, encoding="utf-8")
    created_files.append(".experiment.yaml")

    _ok({
        "slug": slug,
        "path": str(ws),
        "type": exp_type,
        "created_dirs": created_dirs,
        "created_files": created_files,
    })


def cmd_validate(args: argparse.Namespace) -> None:
    """Check workspace structural completeness."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}", suggested_fix=f"Run: python3 tools/exp_workspace.py scaffold --slug {slug}")

    issues = []
    warnings = []

    # Check required directories
    for d in REQUIRED_DIRS:
        if not (ws / d).is_dir():
            issues.append({"type": "missing_dir", "path": d, "severity": "error"})

    # Check required files
    for f, desc in REQUIRED_FILES.items():
        fp = ws / f
        if not fp.exists():
            issues.append({"type": "missing_file", "path": f, "description": desc, "severity": "error"})
        elif fp.stat().st_size == 0:
            warnings.append({"type": "empty_file", "path": f, "severity": "warning"})

    # Check .experiment.yaml
    if not (ws / ".experiment.yaml").exists():
        issues.append({"type": "missing_file", "path": ".experiment.yaml", "severity": "error"})
    else:
        cfg = _read_experiment_yaml(ws)
        if not cfg.get("slug"):
            issues.append({"type": "invalid_config", "detail": "slug is empty in .experiment.yaml", "severity": "error"})

    # Check for Python syntax errors in src/
    src_dir = ws / "src"
    syntax_errors = []
    if src_dir.is_dir():
        for py_file in src_dir.glob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as fh:
                    compile(fh.read(), str(py_file), "exec")
            except SyntaxError as e:
                syntax_errors.append({
                    "file": str(py_file.relative_to(ws)),
                    "line": e.lineno,
                    "message": str(e.msg),
                })

    if syntax_errors:
        for err in syntax_errors:
            issues.append({"type": "syntax_error", "severity": "error", **err})

    valid = len(issues) == 0
    _ok({
        "slug": slug,
        "path": str(ws),
        "valid": valid,
        "issues": issues,
        "warnings": warnings,
        "files_checked": len(list(REQUIRED_FILES.keys())),
    })


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Git commit + tag current workspace state."""
    slug = args.slug
    tag = args.tag
    message = args.message or f"snapshot: {tag}"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    # Initialize git if needed
    git_dir = ws / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=ws, capture_output=True)
        # Create .gitignore
        gitignore = ws / ".gitignore"
        gitignore.write_text(
            "__pycache__/\n*.pyc\n.venv/\nwandb/\ncheckpoints/\n"
            "*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.bin\ndata/\n",
            encoding="utf-8",
        )

    # Stage all changes
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        cwd=ws, capture_output=True, text=True,
    )
    if not result.stdout.strip():
        _ok({"slug": slug, "tag": tag, "committed": False, "reason": "no changes to commit"})

    # Commit
    proc = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=ws, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        _error(f"git commit failed: {proc.stderr.strip()}")

    # Tag
    subprocess.run(
        ["git", "tag", "-f", tag],
        cwd=ws, capture_output=True, text=True,
    )

    # Get commit hash
    hash_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ws, capture_output=True, text=True,
    )
    commit_hash = hash_proc.stdout.strip()[:8] if hash_proc.returncode == 0 else "unknown"

    _ok({
        "slug": slug,
        "tag": tag,
        "committed": True,
        "commit": commit_hash,
        "message": message,
    })


def cmd_diff(args: argparse.Namespace) -> None:
    """Show changes since last snapshot."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if not (ws / ".git").exists():
        _error("Workspace is not git-initialized. Run snapshot first.")

    # Get diff stats
    proc = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=ws, capture_output=True, text=True,
    )
    staged = subprocess.run(
        ["git", "diff", "--stat", "--cached"],
        cwd=ws, capture_output=True, text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ws, capture_output=True, text=True,
    )

    # Changed files
    changed_proc = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=ws, capture_output=True, text=True,
    )
    changed_files = [f for f in changed_proc.stdout.strip().splitlines() if f]

    untracked_files = [f for f in untracked.stdout.strip().splitlines() if f]

    _ok({
        "slug": slug,
        "changed_files": changed_files,
        "untracked_files": untracked_files,
        "diff_summary": proc.stdout.strip(),
        "has_changes": bool(changed_files or untracked_files),
    })


def cmd_config_compose(args: argparse.Namespace) -> None:
    """Merge YAML configs: base + override → composed output."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    base_path = ws / args.base
    if not base_path.exists():
        # Try config/ presets
        root = _find_project_root()
        base_path = root / "config" / "experiment-templates" / args.base
        if not base_path.exists():
            _error(f"Base config not found: {args.base}")

    base = _parse_yaml_simple(base_path.read_text(encoding="utf-8"))

    if args.override:
        override_path = ws / args.override
        if not override_path.exists():
            override_path = _find_project_root() / "config" / "environments" / args.override
        if not override_path.exists():
            _error(f"Override config not found: {args.override}")
        override = _parse_yaml_simple(override_path.read_text(encoding="utf-8"))
        composed = _deep_merge(base, override)
    else:
        composed = base

    _ok({
        "slug": slug,
        "base": args.base,
        "override": args.override or "",
        "composed": composed,
    })


def cmd_status(args: argparse.Namespace) -> None:
    """Workspace health summary."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    # Count files
    total_files = sum(1 for _ in ws.rglob("*") if _.is_file() and ".git" not in _.parts)
    py_files = sum(1 for _ in ws.rglob("*.py") if ".git" not in _.parts)

    # Check git status
    has_git = (ws / ".git").exists()
    dirty = False
    if has_git:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ws, capture_output=True, text=True,
        )
        dirty = bool(proc.stdout.strip())

    # Read state
    cfg = _read_experiment_yaml(ws)
    ws_state = cfg.get("workspace_state", {})

    # Trajectory stats
    traj_dir = ws / ".trajectory"
    events_count = 0
    fixes_count = 0
    if traj_dir.exists():
        events_file = traj_dir / "events.jsonl"
        if events_file.exists():
            events_count = sum(1 for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip())
        fixes_file = traj_dir / "fixes.jsonl"
        if fixes_file.exists():
            fixes_count = sum(1 for line in fixes_file.read_text(encoding="utf-8").splitlines() if line.strip())

    _ok({
        "slug": slug,
        "path": str(ws),
        "phase": ws_state.get("phase", "unknown"),
        "blocked": ws_state.get("blocked", False),
        "total_files": total_files,
        "python_files": py_files,
        "git_initialized": has_git,
        "git_dirty": dirty,
        "events_logged": events_count,
        "fix_rounds": fixes_count,
        "scaffold_strategy": ws_state.get("scaffold_strategy", ""),
    })


def cmd_state(args: argparse.Namespace) -> None:
    """Read workspace_state from .experiment.yaml."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    cfg = _read_experiment_yaml(ws)
    if not cfg:
        _error(".experiment.yaml not found or empty")

    ws_state = cfg.get("workspace_state", {})
    _ok({
        "slug": slug,
        "workspace_state": ws_state,
    })


def cmd_transition(args: argparse.Namespace) -> None:
    """Transition workspace to a new phase (with validation)."""
    slug = args.slug
    to_phase = args.to
    reason = args.reason or ""
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if to_phase not in ALL_PHASES:
        _error(f"Invalid phase: {to_phase}", valid_phases=ALL_PHASES)

    cfg = _read_experiment_yaml(ws)
    if not cfg:
        _error(".experiment.yaml not found or empty")

    ws_state = cfg.get("workspace_state", {})
    current = ws_state.get("phase", "designed")

    # Check valid transition
    allowed = VALID_TRANSITIONS.get(current, [])
    if to_phase not in allowed:
        _error(
            f"Invalid transition: {current} → {to_phase}",
            current_phase=current,
            allowed_transitions=allowed,
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Update state
    ws_state["phase"] = to_phase

    # Record completed phase
    phases_completed = ws_state.get("phases_completed", {})
    if not isinstance(phases_completed, dict):
        phases_completed = {}
    phase_record = {"at": now}
    if reason:
        phase_record["reason"] = reason
    phases_completed[to_phase] = phase_record
    ws_state["phases_completed"] = phases_completed

    # Handle blocked
    if to_phase == "blocked":
        ws_state["blocked"] = True
        ws_state["blocked_reason"] = reason
    else:
        ws_state["blocked"] = False
        ws_state["blocked_reason"] = ""

    cfg["workspace_state"] = ws_state
    _write_experiment_yaml(ws, cfg)

    _ok({
        "slug": slug,
        "previous_phase": current,
        "new_phase": to_phase,
        "reason": reason,
        "timestamp": now,
    })


def cmd_unblock(args: argparse.Namespace) -> None:
    """Clear blocked state, revert to the last completed non-blocked phase."""
    slug = args.slug
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    cfg = _read_experiment_yaml(ws)
    if not cfg:
        _error(".experiment.yaml not found or empty")

    ws_state = cfg.get("workspace_state", {})
    current = ws_state.get("phase", "designed")

    if current != "blocked":
        _error(f"Workspace is not blocked (current phase: {current})")

    # Find the last completed non-blocked phase
    phases_completed = ws_state.get("phases_completed", {})
    # Determine the phase before blocked
    phase_order = [
        "designed", "baseline_checked", "scaffolded", "validated",
        "running", "completed", "collected", "evaluated",
    ]
    revert_to = "designed"
    for phase in reversed(phase_order):
        if phase in phases_completed and phase != "blocked":
            revert_to = phase
            break

    ws_state["phase"] = revert_to
    ws_state["blocked"] = False
    ws_state["blocked_reason"] = ""
    cfg["workspace_state"] = ws_state
    _write_experiment_yaml(ws, cfg)

    _ok({
        "slug": slug,
        "previous_phase": "blocked",
        "reverted_to": revert_to,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment workspace manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # scaffold
    p = sub.add_parser("scaffold", help="Create standard workspace directory")
    p.add_argument("--slug", required=True, help="Experiment slug")
    p.add_argument("--type", default=None, help="Template type (classification/generation/reinforcement/llm-eval)")

    # validate
    p = sub.add_parser("validate", help="Check workspace structural completeness")
    p.add_argument("--slug", required=True, help="Experiment slug")

    # snapshot
    p = sub.add_parser("snapshot", help="Git commit + tag")
    p.add_argument("--slug", required=True, help="Experiment slug")
    p.add_argument("--tag", required=True, help="Snapshot tag name")
    p.add_argument("--message", default=None, help="Commit message")

    # diff
    p = sub.add_parser("diff", help="Show changes since last snapshot")
    p.add_argument("--slug", required=True, help="Experiment slug")

    # config-compose
    p = sub.add_parser("config-compose", help="Merge YAML configs")
    p.add_argument("--slug", required=True, help="Experiment slug")
    p.add_argument("--base", required=True, help="Base config file")
    p.add_argument("--override", default=None, help="Override config file")

    # status
    p = sub.add_parser("status", help="Workspace health summary")
    p.add_argument("--slug", required=True, help="Experiment slug")

    # state
    p = sub.add_parser("state", help="Read workspace_state")
    p.add_argument("--slug", required=True, help="Experiment slug")

    # transition
    p = sub.add_parser("transition", help="State machine transition")
    p.add_argument("--slug", required=True, help="Experiment slug")
    p.add_argument("--to", required=True, help="Target phase")
    p.add_argument("--reason", default=None, help="Transition reason")

    # unblock
    p = sub.add_parser("unblock", help="Clear blocked state")
    p.add_argument("--slug", required=True, help="Experiment slug")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "scaffold": cmd_scaffold,
        "validate": cmd_validate,
        "snapshot": cmd_snapshot,
        "diff": cmd_diff,
        "config-compose": cmd_config_compose,
        "status": cmd_status,
        "state": cmd_state,
        "transition": cmd_transition,
        "unblock": cmd_unblock,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
