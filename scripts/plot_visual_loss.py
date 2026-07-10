import argparse
import os
import re
from typing import List, Tuple

import numpy as np

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LINE_RE = re.compile(rf"Step\[(\d+)/\d+\].*?action_loss:\s*({_FLOAT_RE})")


def parse_action_loss(log_path: str) -> Tuple[np.ndarray, np.ndarray]:
    steps: List[int] = []
    losses: List[float] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            loss = float(m.group(2))
            if not np.isfinite(loss):
                continue
            steps.append(step)
            losses.append(loss)
    if not steps:
        raise ValueError(f"No action_loss found in {log_path}")
    return np.asarray(steps, dtype=np.int64), np.asarray(losses, dtype=np.float64)


def moving_average_trailing(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    window = int(window)
    csum = np.cumsum(y, dtype=np.float64)
    out = np.empty_like(y, dtype=np.float64)
    for i in range(len(y)):
        j0 = max(0, i - window + 1)
        total = csum[i] - (csum[j0 - 1] if j0 > 0 else 0.0)
        out[i] = total / float(i - j0 + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to train log file")
    parser.add_argument("--out", default=None, help="Path to output png")
    parser.add_argument("--window", type=int, default=200, help="Moving average window (trailing)")
    parser.add_argument("--max_points", type=int, default=500000, help="Downsample if too many points")
    parser.add_argument("--min_step", type=int, default=0, help="Only plot points with step >= min_step")
    parser.add_argument("--annotate_every", type=int, default=5000, help="Annotate mean loss every N steps (<=0 to disable)")
    args = parser.parse_args()

    steps, action_loss = parse_action_loss(args.log)
    order = np.argsort(steps)
    steps = steps[order]
    action_loss = action_loss[order]

    keep = steps >= args.min_step
    steps = steps[keep]
    action_loss = action_loss[keep]
    if len(steps) == 0:
        raise ValueError(f"No action_loss points with step >= {args.min_step}")

    steps_full = steps
    action_loss_full = action_loss

    if args.max_points > 0 and len(steps) > args.max_points:
        stride = int(np.ceil(len(steps) / float(args.max_points)))
        steps = steps[::stride]
        action_loss = action_loss[::stride]

    action_loss_smooth = moving_average_trailing(action_loss, window=args.window)

    out_path = args.out
    if out_path is None:
        base = os.path.basename(args.log)
        if base.endswith(".log"):
            base = base[:-4]
        out_path = base + ".action_loss.png"

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4))
    plt.plot(steps, action_loss, linewidth=1.0, alpha=0.25, label="raw")
    plt.plot(steps, action_loss_smooth, linewidth=2.0, alpha=0.95, label=f"ma(window={args.window})")
    plt.axhline(0.005, color="red", linewidth=0.5, alpha=0.9)
    if args.annotate_every > 0:
        every = int(args.annotate_every)
        start_step = int(steps_full[0] // every) * every
        end_step = int(steps_full[-1])
        xs: List[int] = []
        ys: List[float] = []
        for b in range(start_step, end_step + 1, every):
            e = b + every
            m = (steps_full >= b) & (steps_full < e)
            if not np.any(m):
                continue
            mean_loss = float(np.mean(action_loss_full[m]))
            x = int(steps_full[m][-1])
            xs.append(x)
            ys.append(mean_loss)
            plt.text(x, mean_loss, f"{mean_loss:.4f}", fontsize=7, alpha=0.9, rotation=30, ha="left", va="bottom")
        if xs:
            plt.scatter(xs, ys, s=12, alpha=0.9, label=f"mean/ {every} steps")
    plt.xlabel("step")
    plt.ylabel("action_loss")
    plt.ylim(0.0, 0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
