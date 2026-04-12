#!/usr/bin/env python3
"""Execution Environment Abstraction.

Unified interface for running experiments across local, remote-ssh,
and docker backends.  All output is structured JSON following the ACI
principle (LLM-optimized, with error taxonomy and suggested fixes).

Called by skills via:  Bash: python3 tools/exp_env.py <command> [args]

Commands:
    setup     --slug S --backend B [--env-preset P]  Prepare execution environment
    run       --slug S --script R [--config C] [--detach] [--gpu G]  Execute experiment
    check     --slug S                               Check experiment status (structured)
    collect   --slug S [--dest D]                    Collect results from environment
    teardown  --slug S                               Clean up environment resources
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_BASE = "experiments/code"

# Error taxonomy (ACI principle: structured errors with suggested fixes)
ERROR_TAXONOMY = {
    "IMPORT_ERROR": {
        "patterns": [r"ModuleNotFoundError", r"ImportError"],
        "suggested_fix": "pip install {module} or add to requirements.txt",
    },
    "OOM": {
        "patterns": [r"CUDA out of memory", r"RuntimeError:.*allocate", r"OutOfMemoryError"],
        "suggested_fix": "Reduce batch_size, enable gradient_checkpointing, or use mixed precision",
    },
    "NAN_LOSS": {
        "patterns": [r"loss\s*[:=]\s*nan", r"NaN detected"],
        "suggested_fix": "Reduce learning_rate, check data preprocessing, use clip_grad_norm",
    },
    "DATA_ERROR": {
        "patterns": [r"FileNotFoundError.*data", r"No such file or directory.*data"],
        "suggested_fix": "Check data_path in config, ensure dataset is downloaded",
    },
    "SHAPE_MISMATCH": {
        "patterns": [r"size mismatch", r"Expected .* but got", r"RuntimeError:.*shape"],
        "suggested_fix": "Check model config vs data dimensions",
    },
    "TIMEOUT": {
        "patterns": [r"TimeoutError", r"timed out"],
        "suggested_fix": "Reduce epoch count or data size, or increase timeout",
    },
    "CONVERGENCE": {
        "patterns": [r"loss.*plateau", r"early.?stop"],
        "suggested_fix": "Adjust lr/optimizer, check data shuffling",
    },
}

COMPILED_PATTERNS = {
    cat: [(re.compile(p, re.IGNORECASE), cat) for p in info["patterns"]]
    for cat, info in ERROR_TAXONOMY.items()
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "CLAUDE.md").exists() and (p / "tools").is_dir():
            return p
    return cwd


def _workspace_path(slug: str) -> Path:
    return _find_project_root() / WORKSPACE_BASE / slug


def _ok(data: dict) -> None:
    data["status"] = "ok"
    print(json.dumps(data, ensure_ascii=False, indent=2))
    sys.exit(0)


def _error(msg: str, **extra) -> None:
    out = {"status": "error", "message": msg}
    out.update(extra)
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(1)


def _classify_error(text: str) -> dict:
    """Classify error text into taxonomy category with suggested fix."""
    for cat, patterns in COMPILED_PATTERNS.items():
        for regex, _ in patterns:
            m = regex.search(text)
            if m:
                fix = ERROR_TAXONOMY[cat]["suggested_fix"]
                # Try to extract module name for IMPORT_ERROR
                if cat == "IMPORT_ERROR":
                    mod_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
                    if mod_match:
                        fix = fix.format(module=mod_match.group(1))
                return {
                    "category": cat,
                    "matched": m.group(0)[:100],
                    "suggested_fixes": [fix],
                }
    return {
        "category": "UNKNOWN",
        "matched": text[:200] if text else "",
        "suggested_fixes": ["Check full traceback in logs"],
    }


def _read_env_preset(preset_name: str) -> dict:
    """Read environment preset from config/environments/."""
    root = _find_project_root()
    # Try with and without .yaml extension
    for name in [preset_name, f"{preset_name}.yaml"]:
        p = root / "config" / "environments" / name
        if p.exists():
            # Reuse YAML parser from remote.py
            sys.path.insert(0, str(root / "tools"))
            from remote import _parse_yaml
            return _parse_yaml(p.read_text(encoding="utf-8"))
    return {}


def _load_server_config():
    """Load remote server config via remote.py."""
    root = _find_project_root()
    sys.path.insert(0, str(root / "tools"))
    from remote import load_config
    return load_config()


# ---------------------------------------------------------------------------
# Backend: local
# ---------------------------------------------------------------------------


def _local_setup(ws: Path, env_preset: dict) -> dict:
    """Set up local environment: create venv, install deps."""
    req = ws / "requirements.txt"
    if not req.exists():
        return {"backend": "local", "ready": True, "note": "no requirements.txt"}

    # Check if venv exists
    venv = ws / ".venv"
    if not venv.exists():
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return {"backend": "local", "ready": False, "error": f"venv creation failed: {proc.stderr.strip()}"}

    # Install requirements
    pip = str(venv / "bin" / "pip") if os.name != "nt" else str(venv / "Scripts" / "pip.exe")
    proc = subprocess.run(
        [pip, "install", "-q", "-r", str(req)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return {"backend": "local", "ready": False, "error": f"pip install failed: {proc.stderr.strip()[:500]}"}

    return {"backend": "local", "ready": True, "venv": str(venv)}


def _local_run(ws: Path, script: str, config: str | None, detach: bool, gpu: str | None) -> dict:
    """Run experiment locally."""
    script_path = ws / script
    if not script_path.exists():
        # Try scripts/ subdirectory
        script_path = ws / "scripts" / script
    if not script_path.exists():
        return {"ok": False, "error": f"Script not found: {script}"}

    # Determine Python interpreter
    venv = ws / ".venv"
    if (venv / "bin" / "python").exists():
        python = str(venv / "bin" / "python")
    elif (venv / "Scripts" / "python.exe").exists():
        python = str(venv / "Scripts" / "python.exe")
    else:
        python = sys.executable

    # Build command
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if script.endswith(".py"):
        cmd = [python, str(script_path)]
        if config:
            cmd.extend(["--config", config])
    elif script.endswith(".sh"):
        cmd = ["bash", str(script_path)]
        if config:
            env["CONFIG"] = config
    else:
        cmd = [str(script_path)]

    log_dir = ws / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "run.log"

    if detach:
        # Run in background, redirect to log
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, cwd=str(ws), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        # Write PID for later checking
        (ws / ".pid").write_text(str(proc.pid), encoding="utf-8")
        return {
            "ok": True,
            "pid": proc.pid,
            "detached": True,
            "log_file": str(log_file),
        }
    else:
        # Run synchronously
        try:
            proc = subprocess.run(
                cmd, cwd=str(ws), env=env,
                capture_output=True, text=True,
                timeout=3600,
            )
            # Write output to log
            log_file.write_text(proc.stdout + proc.stderr, encoding="utf-8")

            if proc.returncode != 0:
                error_info = _classify_error(proc.stderr or proc.stdout)
                return {
                    "ok": False,
                    "returncode": proc.returncode,
                    "error": error_info,
                    "log_file": str(log_file),
                    "tail": (proc.stderr or proc.stdout).strip().splitlines()[-20:],
                }
            return {
                "ok": True,
                "returncode": 0,
                "detached": False,
                "log_file": str(log_file),
                "tail": proc.stdout.strip().splitlines()[-10:],
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": _classify_error("TimeoutError: experiment exceeded 1 hour"),
                "log_file": str(log_file),
            }


def _local_check(ws: Path) -> dict:
    """Check local experiment status."""
    pid_file = ws / ".pid"
    log_file = ws / "logs" / "run.log"

    if not pid_file.exists():
        # No PID file — check if results exist
        results_dir = ws / "results"
        if results_dir.exists() and any(results_dir.iterdir()):
            return {"alive": False, "exit_reason": "completed", "has_results": True}
        return {"alive": False, "exit_reason": "not_started", "has_results": False}

    pid = int(pid_file.read_text(encoding="utf-8").strip())

    # Check if process is running
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False

    # Read recent log
    tail_lines = []
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        tail_lines = lines[-30:]

    # Detect anomalies
    anomalies = []
    for line in tail_lines:
        for cat, patterns in COMPILED_PATTERNS.items():
            for regex, _ in patterns:
                if regex.search(line):
                    anomalies.append({"type": cat, "content": line.strip()[:200]})
                    break

    # Deduplicate anomalies by type
    seen = set()
    unique_anomalies = []
    for a in anomalies:
        if a["type"] not in seen:
            unique_anomalies.append(a)
            seen.add(a["type"])

    result = {
        "alive": alive,
        "pid": pid,
        "last_lines": tail_lines[-10:],
        "anomalies": unique_anomalies,
    }

    if not alive:
        result["exit_reason"] = "crashed" if unique_anomalies else "completed"

    return result


def _local_collect(ws: Path, dest: str | None) -> dict:
    """Collect results from local workspace."""
    results_dir = ws / "results"
    logs_dir = ws / "logs"

    if dest:
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        # Copy results
        import shutil
        if results_dir.exists():
            shutil.copytree(results_dir, dest_path / "results", dirs_exist_ok=True)
        if logs_dir.exists():
            shutil.copytree(logs_dir, dest_path / "logs", dirs_exist_ok=True)

    # Parse result files
    result_files = []
    metrics = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            result_files.append(str(f.relative_to(ws)))
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            if k not in metrics:
                                metrics[k] = []
                            metrics[k].append(v)
            except (json.JSONDecodeError, ValueError):
                pass

    # Aggregate metrics
    aggregated = {}
    for k, values in metrics.items():
        if values:
            aggregated[k] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return {
        "result_files": result_files,
        "metrics": aggregated,
        "logs_available": logs_dir.exists() and any(logs_dir.iterdir()) if logs_dir.exists() else False,
    }


def _local_teardown(ws: Path) -> dict:
    """Clean up local environment (remove venv, pid file)."""
    cleaned = []
    pid_file = ws / ".pid"
    if pid_file.exists():
        pid_file.unlink()
        cleaned.append(".pid")

    venv = ws / ".venv"
    if venv.exists():
        import shutil
        shutil.rmtree(venv)
        cleaned.append(".venv")

    return {"cleaned": cleaned}


# ---------------------------------------------------------------------------
# Backend: remote-ssh (delegates to remote.py)
# ---------------------------------------------------------------------------


def _remote_setup(ws: Path, slug: str, env_preset: dict) -> dict:
    """Set up remote environment via remote.py."""
    root = _find_project_root()
    sys.path.insert(0, str(root / "tools"))
    try:
        from remote import load_config, run_ssh, conda_prefix, build_ssh_cmd
        cfg = load_config()
    except Exception as e:
        return {"backend": "remote-ssh", "ready": False, "error": f"Cannot load server config: {e}"}

    # Check connectivity
    rc, _, stderr = run_ssh(cfg, "echo ok", timeout=10)
    if rc != 0:
        return {"backend": "remote-ssh", "ready": False, "error": f"SSH connection failed: {stderr.strip()}"}

    # Check GPU availability
    rc, stdout, _ = run_ssh(
        cfg,
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
        timeout=15,
    )
    free_gpus = []
    if rc == 0:
        for line in stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    idx, used, total = int(parts[0]), int(float(parts[1])), int(float(parts[2]))
                    if used < 500:
                        free_gpus.append(idx)
                except ValueError:
                    pass

    return {
        "backend": "remote-ssh",
        "ready": True,
        "host": cfg.get("host", ""),
        "free_gpus": free_gpus,
        "gpu_memory_free_mb": [],
    }


def _remote_run(ws: Path, slug: str, script: str, config: str | None, detach: bool, gpu: str | None) -> dict:
    """Run experiment on remote via remote.py."""
    root = _find_project_root()
    sys.path.insert(0, str(root / "tools"))
    try:
        from remote import load_config, run_ssh, conda_prefix
    except Exception as e:
        return {"ok": False, "error": f"Cannot load remote module: {e}"}

    cfg = load_config()

    # Sync code first
    sync_result = subprocess.run(
        [sys.executable, str(root / "tools" / "remote.py"), "sync-code", "--local-path", str(ws)],
        capture_output=True, text=True,
    )
    if sync_result.returncode != 0:
        return {"ok": False, "error": f"Code sync failed: {sync_result.stdout}"}

    # Install dependencies
    req = ws / "requirements.txt"
    if req.exists():
        setup_result = subprocess.run(
            [sys.executable, str(root / "tools" / "remote.py"), "setup-env", "--requirements", str(req)],
            capture_output=True, text=True,
        )

    if detach:
        # Launch via remote.py screen session
        cmd_parts = [
            sys.executable, str(root / "tools" / "remote.py"),
            "launch", "--name", f"exp-{slug}", "--cmd", f"bash scripts/{script}" if script.endswith(".sh") else f"python {script}",
        ]
        if gpu is not None:
            cmd_parts.extend(["--gpu", str(gpu)])

        launch_result = subprocess.run(cmd_parts, capture_output=True, text=True)
        try:
            data = json.loads(launch_result.stdout)
            return {"ok": data.get("status") == "ok", "detached": True, **data}
        except json.JSONDecodeError:
            return {"ok": False, "error": launch_result.stdout or launch_result.stderr}
    else:
        # Synchronous remote run
        prefix = conda_prefix(cfg)
        work_dir = cfg["work_dir"]
        gpu_part = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu else ""
        if script.endswith(".sh"):
            remote_cmd = f"cd {shlex.quote(work_dir)} && {prefix} && {gpu_part}bash scripts/{script}"
        else:
            remote_cmd = f"cd {shlex.quote(work_dir)} && {prefix} && {gpu_part}python {script}"

        if config:
            remote_cmd += f" --config {config}"

        rc, stdout, stderr = run_ssh(cfg, remote_cmd, timeout=3600)
        if rc != 0:
            error_info = _classify_error(stderr or stdout)
            return {"ok": False, "returncode": rc, "error": error_info, "tail": (stderr or stdout).splitlines()[-20:]}
        return {"ok": True, "returncode": 0, "detached": False, "tail": stdout.splitlines()[-10:]}


def _remote_check(ws: Path, slug: str) -> dict:
    """Check remote experiment status."""
    root = _find_project_root()
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "remote.py"), "check", "--name", f"exp-{slug}"],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        return data
    except json.JSONDecodeError:
        return {"alive": False, "error": result.stdout or result.stderr}


def _remote_collect(ws: Path, slug: str, dest: str | None) -> dict:
    """Collect results from remote."""
    root = _find_project_root()
    local_dest = dest or str(ws / "results")
    Path(local_dest).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, str(root / "tools" / "remote.py"),
            "pull-results", "--remote-path", "results/", "--local-path", local_dest,
        ],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        return data
    except json.JSONDecodeError:
        return {"error": result.stdout or result.stderr}


def _remote_teardown(ws: Path, slug: str) -> dict:
    """Teardown remote resources (no-op currently, screen auto-exits)."""
    return {"cleaned": [], "note": "Remote screen sessions auto-exit on completion"}


# ---------------------------------------------------------------------------
# Backend: docker (stub)
# ---------------------------------------------------------------------------


def _docker_stub(operation: str) -> dict:
    return {
        "backend": "docker",
        "error": f"Docker backend not yet implemented (operation: {operation})",
        "suggested_fix": "Use --backend local or --backend remote-ssh",
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    """Prepare execution environment."""
    slug = args.slug
    backend = args.backend or "local"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}", suggested_fix=f"Run: python3 tools/exp_workspace.py scaffold --slug {slug}")

    env_preset = _read_env_preset(args.env_preset) if args.env_preset else {}

    if backend == "local":
        result = _local_setup(ws, env_preset)
    elif backend in ("remote-ssh", "remote"):
        result = _remote_setup(ws, slug, env_preset)
    elif backend == "docker":
        result = _docker_stub("setup")
    else:
        _error(f"Unknown backend: {backend}", valid_backends=["local", "remote-ssh", "docker"])
        return

    result["backend"] = backend
    if result.get("ready", False) or result.get("error"):
        if "error" in result and not result.get("ready", True):
            _error(result["error"], **{k: v for k, v in result.items() if k != "error"})
        _ok(result)
    else:
        _ok(result)


def cmd_run(args: argparse.Namespace) -> None:
    """Execute experiment."""
    slug = args.slug
    backend = args.backend or "local"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if backend == "local":
        result = _local_run(ws, args.script, args.config, args.detach, args.gpu)
    elif backend in ("remote-ssh", "remote"):
        result = _remote_run(ws, slug, args.script, args.config, args.detach, args.gpu)
    elif backend == "docker":
        result = _docker_stub("run")
    else:
        _error(f"Unknown backend: {backend}")
        return

    result["backend"] = backend
    result["slug"] = slug
    if result.get("ok", False):
        _ok(result)
    else:
        out = {"status": "error", "message": result.get("error", "Run failed")}
        out.update(result)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(1)


def cmd_check(args: argparse.Namespace) -> None:
    """Check experiment status."""
    slug = args.slug
    backend = args.backend or "local"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if backend == "local":
        result = _local_check(ws)
    elif backend in ("remote-ssh", "remote"):
        result = _remote_check(ws, slug)
    elif backend == "docker":
        result = _docker_stub("check")
    else:
        _error(f"Unknown backend: {backend}")
        return

    result["backend"] = backend
    result["slug"] = slug
    _ok(result)


def cmd_collect(args: argparse.Namespace) -> None:
    """Collect results from environment."""
    slug = args.slug
    backend = args.backend or "local"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if backend == "local":
        result = _local_collect(ws, args.dest)
    elif backend in ("remote-ssh", "remote"):
        result = _remote_collect(ws, slug, args.dest)
    elif backend == "docker":
        result = _docker_stub("collect")
    else:
        _error(f"Unknown backend: {backend}")
        return

    result["backend"] = backend
    result["slug"] = slug
    _ok(result)


def cmd_teardown(args: argparse.Namespace) -> None:
    """Clean up environment resources."""
    slug = args.slug
    backend = args.backend or "local"
    ws = _workspace_path(slug)

    if not ws.exists():
        _error(f"Workspace not found: {ws}")

    if backend == "local":
        result = _local_teardown(ws)
    elif backend in ("remote-ssh", "remote"):
        result = _remote_teardown(ws, slug)
    elif backend == "docker":
        result = _docker_stub("teardown")
    else:
        _error(f"Unknown backend: {backend}")
        return

    result["backend"] = backend
    result["slug"] = slug
    _ok(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution environment abstraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # setup
    p = sub.add_parser("setup", help="Prepare execution environment")
    p.add_argument("--slug", required=True)
    p.add_argument("--backend", default="local", choices=["local", "remote-ssh", "remote", "docker"])
    p.add_argument("--env-preset", default=None, help="Environment preset name from config/environments/")

    # run
    p = sub.add_parser("run", help="Execute experiment")
    p.add_argument("--slug", required=True)
    p.add_argument("--script", default="scripts/run.sh", help="Script to execute")
    p.add_argument("--config", default=None, help="Config file to pass")
    p.add_argument("--detach", action="store_true", help="Run in background")
    p.add_argument("--gpu", default=None, help="GPU index (e.g. 0 or 0,1)")
    p.add_argument("--backend", default="local", choices=["local", "remote-ssh", "remote", "docker"])

    # check
    p = sub.add_parser("check", help="Check experiment status")
    p.add_argument("--slug", required=True)
    p.add_argument("--backend", default="local", choices=["local", "remote-ssh", "remote", "docker"])

    # collect
    p = sub.add_parser("collect", help="Collect results")
    p.add_argument("--slug", required=True)
    p.add_argument("--dest", default=None, help="Local destination directory")
    p.add_argument("--backend", default="local", choices=["local", "remote-ssh", "remote", "docker"])

    # teardown
    p = sub.add_parser("teardown", help="Clean up environment")
    p.add_argument("--slug", required=True)
    p.add_argument("--backend", default="local", choices=["local", "remote-ssh", "remote", "docker"])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "setup": cmd_setup,
        "run": cmd_run,
        "check": cmd_check,
        "collect": cmd_collect,
        "teardown": cmd_teardown,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
