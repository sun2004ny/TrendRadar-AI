#!/usr/bin/env python3
"""
TrendRadar AI - Application Launcher
====================================
Main entry point for starting the TrendRadar AI Intelligence Platform.

Usage:
    python run.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    app_script = root_dir / "app" / "streamlit_app.py"

    if not app_script.exists():
        print(f"Error: Could not find application script at {app_script}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  TrendRadar AI Starting...")
    print("  Real-Time Topic Emergence & Intelligence Platform")
    print("  Dashboard: http://localhost:8501")
    print("=" * 60)
    print(f"Interpreter : {sys.executable}")
    print(f"Application : {app_script}")
    print("\nStarting Streamlit server... (Press Ctrl+C to stop)\n")

    # Command: python -m streamlit run app/streamlit_app.py
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_script),
        "--server.headless=false",
    ]

    # Forward any additional command line flags if provided
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Configure environment for clean UTF-8 output on Windows
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        # Launch Streamlit using module execution with the current interpreter
        process = subprocess.run(cmd, cwd=str(root_dir), env=env)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\n[TrendRadar AI] Dashboard server stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
