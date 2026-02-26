#!/usr/bin/env python3
"""
Visualize beam-search evolution from a decoder CSV log.

Modes:
  terminal  - Animate in the terminal (clear + redraw each step).
  gif       - Export an animated GIF (requires Pillow).
  mp4       - Export MP4 video (requires ffmpeg or pillow; may fall back to gif).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def load_beam_csv(path: str | Path) -> list[dict]:
    """Load beam log CSV and return list of row dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["rank"] = int(row["rank"])
            row["last_frame"] = int(row["last_frame"])
            row["score"] = float(row["score"])
            row["finished"] = row["finished"].strip().lower() in ("true", "1", "yes")
            row["last_am_lp"] = float(row["last_am_lp"])
            row["last_lm_lp"] = float(row["last_lm_lp"])
            rows.append(row)
    return rows


def group_by_step(rows: list[dict]) -> dict[int, list[dict]]:
    """Group rows by step, each value sorted by rank."""
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        s = row["step"]
        by_step.setdefault(s, []).append(row)
    for s in by_step:
        by_step[s].sort(key=lambda r: r["rank"])
    return by_step


def run_terminal_animation(
    by_step: dict[int, list[dict]],
    step_delay: float = 0.8,
    clear_screen: bool = True,
) -> None:
    """Print beam state step-by-step in the terminal."""
    steps = sorted(by_step.keys())
    if not steps:
        print("No steps in CSV.")
        return

    def clear():
        if clear_screen:
            print("\033[2J\033[H", end="")  # ANSI: clear and home
        else:
            print("\n" + "=" * 72 + "\n")

    try:
        for step in steps:
            clear()
            rows = by_step[step]
            print(f"  Step {step}  (beam size = {len(rows)})")
            print("-" * 88)
            # Header (include AM/LM scores)
            print(f"  {'rank':<4} {'score':>8} {'am':>8} {'lm':>8} {'fin':<3}  text")
            print("-" * 88)
            for r in rows:
                fin = "yes" if r["finished"] else "no"
                text = (r["text"] or "(empty)").replace("\n", " ")[:40]
                if len(r["text"]) > 40:
                    text += "..."
                print(
                    f"  {r['rank']:<4} "
                    f"{r['score']:>8.4f} "
                    f"{r['last_am_lp']:>8.4f} "
                    f"{r['last_lm_lp']:>8.4f} "
                    f"{fin:<3}  {text}"
                )
            print("-" * 88)
            sys.stdout.flush()
            time.sleep(step_delay)
        clear()
        print("  Done.")
    except KeyboardInterrupt:
        clear()
        print("  Stopped.")


def _draw_beam_step(ax, by_step: dict[int, list[dict]], step: int) -> None:
    """Draw a single step's beam table on ax (dark theme)."""
    ax.clear()
    ax.axis("off")
    ax.set_facecolor("#1a1a1a")
    rows = by_step[step]
    ax.set_title(f"Beam search — step {step}", fontsize=14, color="#e0e0e0", pad=12)
    y = 1.0
    line_height = 0.08
    ax.text(0.02, y, "rank", fontsize=10, color="#888", family="monospace")
    ax.text(0.10, y, "score", fontsize=10, color="#888", family="monospace")
    ax.text(0.26, y, "am", fontsize=10, color="#888", family="monospace")
    ax.text(0.38, y, "lm", fontsize=10, color="#888", family="monospace")
    ax.text(0.50, y, "done", fontsize=10, color="#888", family="monospace")
    ax.text(0.60, y, "text", fontsize=10, color="#888", family="monospace")
    y -= line_height
    for r in rows:
        fin = "✓" if r["finished"] else ""
        text = (r["text"] or "(empty)").replace("\n", " ")[:42]
        ax.text(0.02, y, str(r["rank"]), fontsize=10, color="#fff", family="monospace")
        ax.text(0.10, y, f"{r['score']:.2f}", fontsize=10, color="#aaa", family="monospace")
        ax.text(0.26, y, f"{r['last_am_lp']:.2f}", fontsize=10, color="#88c", family="monospace")
        ax.text(0.38, y, f"{r['last_lm_lp']:.2f}", fontsize=10, color="#c88", family="monospace")
        ax.text(0.50, y, fin, fontsize=10, color="#6a6", family="monospace")
        ax.text(0.60, y, text, fontsize=10, color="#e0e0e0", family="monospace")
        y -= line_height
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.6)


def run_gif_export(
    by_step: dict[int, list[dict]],
    output_path: str | Path,
    step_duration_sec: float = 0.6,
    dpi: int = 100,
) -> None:
    """Render beam steps as an animated GIF (requires Pillow)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as mplanim
    except ImportError as e:
        print("matplotlib required for GIF/MP4 export.", file=sys.stderr)
        raise SystemExit(1) from e

    # GIF saving uses matplotlib's 'pillow' writer (requires Pillow)
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Pillow required for GIF export. Install with: pip install Pillow", file=sys.stderr)
        raise SystemExit(1)

    steps = sorted(by_step.keys())
    if not steps:
        print("No steps in CSV.")
        return

    step_list = list(steps)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")

    def frame(i: int):
        _draw_beam_step(ax, by_step, step_list[i])

    anim = mplanim.FuncAnimation(
        fig,
        frame,
        frames=len(step_list),
        interval=step_duration_sec * 1000,
        repeat=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(1 / step_duration_sec))
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_mp4_export(
    by_step: dict[int, list[dict]],
    output_path: str | Path,
    step_duration_sec: float = 0.6,
    fps: int = 2,
) -> None:
    """Export beam animation as MP4 using matplotlib animation (needs ffmpeg or pillow)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as mplanim
    except ImportError as e:
        print("matplotlib required for MP4 export.", file=sys.stderr)
        raise SystemExit(1) from e

    steps = sorted(by_step.keys())
    if not steps:
        print("No steps in CSV.")
        return

    step_list = list(steps)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")

    def frame(i: int):
        _draw_beam_step(ax, by_step, step_list[i])

    anim = mplanim.FuncAnimation(
        fig,
        frame,
        frames=len(step_list),
        interval=step_duration_sec * 1000,
        repeat=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=100)
    except Exception:
        try:
            anim.save(str(output_path.with_suffix(".gif")), writer="pillow", fps=fps, dpi=100)
            print("ffmpeg not available; saved as GIF instead:", output_path.with_suffix(".gif"))
        except Exception as e:
            print("Could not save MP4 or GIF:", e, file=sys.stderr)
            raise SystemExit(1) from e
    else:
        print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize beam-search evolution from decoder CSV log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to beam log CSV. If omitted, use latest in beam_logs/.",
    )
    parser.add_argument(
        "--mode",
        choices=("terminal", "gif", "mp4"),
        default="terminal",
        help="terminal: animate in terminal; gif/mp4: export video file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Seconds per step in terminal mode.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.6,
        help="Seconds per step in GIF/MP4 (frame duration).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for gif/mp4 (default: beam_logs/beam_viz.gif or .mp4).",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="In terminal mode, do not clear screen between steps (scroll instead).",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        beam_dir = Path("beam_logs")
        if not beam_dir.is_dir():
            print("No beam_logs/ directory and no CSV path given.", file=sys.stderr)
            sys.exit(1)
        files = sorted(beam_dir.glob("beam_*.csv"), key=os.path.getmtime, reverse=True)
        if not files:
            print("No beam_*.csv files in beam_logs/.", file=sys.stderr)
            sys.exit(1)
        csv_path = files[0]
        print(f"Using latest log: {csv_path}")

    if not csv_path.is_file():
        print(f"Not a file: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_beam_csv(csv_path)
    by_step = group_by_step(rows)
    print(f"Loaded {len(rows)} rows, {len(by_step)} steps.")

    if args.mode == "terminal":
        run_terminal_animation(
            by_step,
            step_delay=args.delay,
            clear_screen=not args.no_clear,
        )
    elif args.mode == "gif":
        out = args.output or csv_path.with_suffix(".gif")
        run_gif_export(by_step, out, step_duration_sec=args.duration)
    else:
        out = args.output or csv_path.with_suffix(".mp4")
        run_mp4_export(by_step, out, step_duration_sec=args.duration, fps=max(1, int(1 / args.duration)))


if __name__ == "__main__":
    main()
