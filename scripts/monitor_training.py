#!/usr/bin/env python3
"""
Live monitoring script for training progress.

Continuously displays the latest training status from log files.

Usage:
    python scripts/monitor_training.py outputs/vt_former_pretrain

    # Or with auto-refresh
    python scripts/monitor_training.py outputs/vt_former_pretrain --refresh 5
"""

import os
import sys
import time
import argparse
from pathlib import Path


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def read_progress(progress_file: str) -> str:
    """Read progress.txt file."""
    try:
        with open(progress_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Progress file not found. Training may not have started yet."


def read_latest_log(log_dir: str, n_lines: int = 50) -> str:
    """Read last N lines from the most recent log file."""
    try:
        log_files = list(Path(log_dir).glob("training_*.log"))
        if not log_files:
            return "No log files found yet."

        # Get most recent log file
        latest_log = max(log_files, key=os.path.getctime)

        with open(latest_log, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n_lines:])
    except Exception as e:
        return f"Error reading log: {e}"


def monitor_training(output_dir: str, refresh_interval: int = 0):
    """
    Monitor training progress.

    Args:
        output_dir: Path to training output directory
        refresh_interval: Seconds between refreshes (0 = no auto-refresh)
    """
    progress_file = os.path.join(output_dir, "progress.txt")
    log_dir = os.path.join(output_dir, "logs")

    try:
        while True:
            if refresh_interval > 0:
                clear_screen()

            print("=" * 80)
            print("WormVT Training Monitor")
            print("=" * 80)
            print(f"Output Directory: {output_dir}")
            print()

            # Show progress summary
            print("--- PROGRESS SUMMARY ---")
            print(read_progress(progress_file))
            print()

            # Show recent log lines
            print("--- RECENT LOG (last 30 lines) ---")
            print(read_latest_log(log_dir, n_lines=30))
            print()

            if refresh_interval == 0:
                break

            print(f"[Refreshing every {refresh_interval}s. Press Ctrl+C to stop]")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def tail_log(log_file: str):
    """Tail a log file (like 'tail -f')."""
    print(f"Tailing {log_file}...")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    try:
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped tailing.")


def main():
    parser = argparse.ArgumentParser(description="Monitor WormVT training progress")
    parser.add_argument(
        "output_dir",
        help="Path to training output directory (e.g., outputs/vt_former_pretrain)"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=0,
        help="Auto-refresh interval in seconds (0 = no auto-refresh)"
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Tail the latest log file (like 'tail -f')"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: Directory not found: {args.output_dir}")
        sys.exit(1)

    if args.tail:
        log_dir = os.path.join(args.output_dir, "logs")
        log_files = list(Path(log_dir).glob("training_*.log"))
        if not log_files:
            print("No log files found yet.")
            sys.exit(1)
        latest_log = max(log_files, key=os.path.getctime)
        tail_log(latest_log)
    else:
        monitor_training(args.output_dir, args.refresh)


if __name__ == "__main__":
    main()
