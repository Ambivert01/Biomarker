#!/usr/bin/env python3
"""
start.py — one-command setup + launch for the ADNI Plasma Biomarker AD Classifier.

What this does, in order:
  1. Checks your Python version.
  2. Installs every required package (pandas, scikit-learn, streamlit, etc.)
  3. Opens the clinician-facing prediction app in your browser.

Usage (from a terminal, inside this project folder):
    python start.py
    (or, on some systems:  python3 start.py)

That's it — no other setup needed.
"""
import subprocess
import sys
import os

MIN_PYTHON = (3, 10)


def banner(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def check_python_version():
    banner("Step 1/3 — Checking your Python version")
    if sys.version_info < MIN_PYTHON:
        print(f"❌ This project needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
        print(f"   You have Python {sys.version_info.major}.{sys.version_info.minor}.")
        print("   Please install a newer Python from https://python.org and try again.")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} — OK")


def install_requirements():
    banner("Step 2/3 — Installing required packages (this can take a few minutes the first time)")
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    except subprocess.CalledProcessError:
        print("\n❌ Something went wrong installing packages. Common fixes:")
        print("   - Make sure you have internet access.")
        print("   - Try running:  python -m pip install --upgrade pip")
        print("   - Then run this script again.")
        sys.exit(1)
    print("✅ All packages installed.")


def launch_app():
    banner("Step 3/3 — Launching the app in your browser")
    print("A browser tab should open automatically at http://localhost:8501")
    print("If it doesn't, just open that address manually.")
    print("\nTo stop the app later, come back to this window and press Ctrl+C.\n")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "streamlit", "app.py")
    subprocess.call([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    check_python_version()
    install_requirements()
    launch_app()
