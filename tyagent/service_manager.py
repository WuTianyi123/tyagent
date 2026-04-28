"""Systemd service manager for tyagent gateway.

Provides install/start/stop/status/uninstall commands for systemd user services.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "tyagent-gateway"
SERVICE_DESCRIPTION = "tyagent messaging gateway"


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def _get_venv_dir() -> Path | None:
    """Detect virtual environment directory."""
    project_root = _get_project_root()
    candidates = [
        project_root / ".venv",
        project_root / "venv",
    ]
    for c in candidates:
        if c.exists() and (c / "bin" / "python").exists():
            return c
    return None


def _get_python_path() -> str:
    """Return the Python interpreter path to use."""
    venv = _get_venv_dir()
    if venv and (venv / "bin" / "python").exists():
        return str(venv / "bin" / "python")
    return shutil.which("python3") or shutil.which("python") or sys.executable


def _get_unit_path() -> Path:
    """Return the systemd user service unit file path."""
    return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"


def _run_systemctl(args: list[str], check: bool = False, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run systemctl with the given args."""
    cmd = ["systemctl", "--user"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def _supports_systemd() -> bool:
    """Check if systemd user services are supported."""
    if not shutil.which("systemctl"):
        return False
    try:
        result = _run_systemctl(["is-system-running"], timeout=5)
        return result.stdout.strip() in ("running", "degraded", "starting")
    except Exception:
        return False


def _generate_unit() -> str:
    """Generate the systemd unit file content."""
    python_path = _get_python_path()
    project_root = _get_project_root()
    venv = _get_venv_dir()
    venv_dir = str(venv) if venv else ""
    
    path_entries = [str(project_root)]
    if venv:
        path_entries.append(str(venv / "bin"))
    common_paths = ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    path_entries.extend(common_paths)
    
    env_lines = []
    for key, value in os.environ.items():
        # Exclude HOME — gateway will set it to profile_home at runtime
        if key in ("PATH", "USER", "OPENAI_API_KEY", "KIMI_API_KEY", 
                    "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
            env_lines.append(f'Environment="{key}={value}"')
    
    # Also add VIRTUAL_ENV if we detected one
    if venv_dir:
        env_lines.append(f'Environment="VIRTUAL_ENV={venv_dir}"')
    
    env_block = '\n'.join(env_lines)
    
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
ExecStart={python_path} -m tyagent_cli gateway
WorkingDirectory={project_root}
Environment="PATH={':'.join(path_entries)}"
{env_block}
Restart=on-failure
RestartSec=10
RestartForceExitStatus=75
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def install_service(force: bool = False) -> int:
    """Install tyagent gateway as a systemd user service."""
    if not _supports_systemd():
        print("Error: systemd is not available or not running.")
        print("You can still run 'tyagent gateway' in the foreground.")
        return 1
    
    unit_path = _get_unit_path()
    
    if unit_path.exists() and not force:
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return 0
    
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_generate_unit(), encoding="utf-8")
    
    _run_systemctl(["daemon-reload"], check=True, timeout=30)
    _run_systemctl(["enable", SERVICE_NAME], check=True, timeout=30)
    
    print(f"✓ Service installed: {unit_path}")
    print()
    print("Commands:")
    print(f"  tyagent gateway start    # Start the service")
    print(f"  tyagent gateway stop     # Stop the service")
    print(f"  tyagent gateway status   # Check status")
    print(f"  journalctl --user -u {SERVICE_NAME} -f  # View logs")
    print()
    
    # Check linger
    username = getpass.getuser()
    linger_file = Path(f"/var/lib/systemd/linger/{username}")
    if not linger_file.exists():
        print("⚠ Note: systemd 'linger' is not enabled.")
        print("  The service may stop when you log out.")
        print(f"  Run 'sudo loginctl enable-linger {username}' to keep it running.")
        print()
    
    return 0


def uninstall_service() -> int:
    """Uninstall the systemd user service."""
    if not _supports_systemd():
        print("Error: systemd is not available.")
        return 1
    
    _run_systemctl(["stop", SERVICE_NAME], check=False, timeout=30)
    _run_systemctl(["disable", SERVICE_NAME], check=False, timeout=30)
    
    unit_path = _get_unit_path()
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")
    
    _run_systemctl(["daemon-reload"], check=True, timeout=30)
    print("✓ Service uninstalled")
    return 0


def start_service() -> int:
    """Start the systemd user service."""
    if not _supports_systemd():
        print("Error: systemd is not available.")
        return 1
    
    unit_path = _get_unit_path()
    if not unit_path.exists():
        print("Service not installed. Run 'tyagent gateway install' first.")
        return 1
    
    _run_systemctl(["start", SERVICE_NAME], check=True, timeout=30)
    print(f"✓ Service started: {SERVICE_NAME}")
    return 0


def stop_service() -> int:
    """Stop the systemd user service."""
    if not _supports_systemd():
        print("Error: systemd is not available.")
        return 1
    
    _run_systemctl(["stop", SERVICE_NAME], check=False, timeout=30)
    print(f"✓ Service stopped: {SERVICE_NAME}")
    return 0


def restart_service() -> int:
    """Restart the systemd user service."""
    if not _supports_systemd():
        print("Error: systemd is not available.")
        return 1
    
    unit_path = _get_unit_path()
    if not unit_path.exists():
        print("Service not installed. Run 'tyagent gateway install' first.")
        return 1
    
    _run_systemctl(["restart", SERVICE_NAME], check=True, timeout=30)
    print(f"✓ Service restarted: {SERVICE_NAME}")
    return 0


def get_pid() -> int | None:
    """Get the PID of the running service, or None if not running."""
    result = _run_systemctl(
        ["show", "--property=MainPID", "--value", SERVICE_NAME],
        timeout=5,
    )
    pid_str = result.stdout.strip()
    try:
        pid = int(pid_str)
        return pid if pid > 0 else None
    except (ValueError, TypeError):
        return None


def status_service() -> int:
    """Show the systemd user service status."""
    if not _supports_systemd():
        print("Error: systemd is not available.")
        return 1
    
    unit_path = _get_unit_path()
    installed = unit_path.exists()
    
    try:
        result = _run_systemctl(["is-active", SERVICE_NAME], timeout=5)
        active = result.stdout.strip() == "active"
    except Exception:
        active = False
    
    print(f"Service:        {SERVICE_NAME}")
    print(f"Installed:      {'Yes' if installed else 'No'} ({unit_path})")
    print(f"Active:         {'Yes' if active else 'No'}")
    print()
    
    if installed:
        print(f"Logs: journalctl --user -u {SERVICE_NAME} -f")
    
    return 0
