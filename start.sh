#!/usr/bin/env bash
# Double-click-friendly launcher for Mac/Linux.
# If double-clicking doesn't work on your system, open a terminal in this
# folder and run:  bash start.sh
cd "$(dirname "$0")"
if command -v python3 &>/dev/null; then
    python3 start.py
elif command -v python &>/dev/null; then
    python start.py
else
    echo "Python was not found on this computer."
    echo "Please install it from https://python.org and try again."
    read -p "Press Enter to close..."
fi
