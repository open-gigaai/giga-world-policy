#!/usr/bin/env python3
"""Compute WAM finetune norm stats over the FULL dataset, matching the live dataloader.

Unlike compute_wam_task_norm.py (segment-manifest oriented, trims episode tails and
reads contiguous future frames), this script replicates FastLeRobotDataset semantics:

- Every frame in [ep_start, ep_end) is a sample start (no valid-window trimming).
- The action horizon is clamped within the episode: idx = min(ep_end - 1, start + h),
  i.e. the last frame is repeated, never crossing into the next episode.
- delta = action[clamped] - state[start] for delta-mask dims, absolute action otherwise.
- Reads parquet only; never decodes video.
- Accepts multiple --data-root values; stats are pooled across all roots into one output
  (matching a cfg norm entry that maps one norm file to a list of data_paths).

Performance (techniques borrowed from fast_norm_stats.py):
- Reads only the needed columns and never decodes video.
- Streams data file-by-file with a small carry-over buffer, so peak memory stays at
  roughly one parquet file plus the in-progress episode (no full-dataset load). This
  correctly handles episodes that span parquet file boundaries.
- Uses float32 buffers for the heavy (n_starts * horizon, dim) delta array to halve
  memory bandwidth; moments still accumulate in float64 for numerical stability, and
  quantiles use reservoir sampling to bound cost.

Stats are emitted at model_dim (default 32) with mean/std/min/max/q01/q99, matching the
existing norm_stats JSON format consumed by WALeRobotTransformsPretrain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# delta_mask from the finetune cfg (embodiment 0). True dims use action-state delta.
DEFAULT_DELTA_MASK = (
    True, True, True, True, True, True, False,
    True, True, True, True, True, True, False,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, nargs="+",
                   help="One or more LeRobot dataset roots (each contains data/ and meta/). "
                        "Stats are pooled across all roots into a single output.")
    p.add_argument("--output", required=True)
    p.add_argument("--task-name", default="finetune")
    p.add_argument("--model-dim", type=int, default=32)
    p.add_argument("--action-horizon", type=int, default=48, help="Matches num_frames / delta_info['action'].")
    p.add_argument("--quantile-sample-limit", type=int, default=2_000_000,
                   help="Reservoir cap per feature for q01/q99. 0 keeps all rows.")
    p.add_argument("--float64", action="store_true",
                   help="Use float64 buffers instead of the default float32 (slower, more memory).")
    p.add_argument("--strict-align", action="store_true",
                   help="Raise instead of warning when action/state are misaligned on delta-mask dims.")
    p.add_argument("--include-metadata", action="store_true")
    return p.parse_args()


class OnlineMoments:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.count += int(x.shape[0])
        self.sum += x.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))

    def finish(self):
        if self.count <= 0:
            raise ValueError("No samples accumulated")
        mean = self.sum / float(self.count)
        var = self.sumsq / float(self.count) - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std, self.min, self.max


class QuantileStore:
    def __init__(self, dim: int, sample_limit: int, seed: int) -> None:
        self.dim = int(dim)
        self.sample_limit = int(sample_limit)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.filled = 0
        self.chunks: list[np.ndarray] = []
        if self.sample_limit > 0:
            self.sample = np.empty((self.sample_limit, dim), dtype=np.float32)
        else:
            self.sample = np.empty((0, dim), dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x32 = x.astype(np.float32, copy=False)
        if self.sample_limit <= 0:
            self.chunks.append(x32.copy())
            self.count += int(x32.shape[0])
            return
        n = int(x32.shape[0])
        start = 0
        if self.filled < self.sample_limit:
            take = min(n, self.sample_limit - self.filled)
            self.sample[self.filled:self.filled + take] = x32[:take]
            self.filled += take
            self.count += take
            start = take
        remaining = n - start
        if remaining <= 0:
            return
        global_counts = np.arange(self.count + 1, self.count + remaining + 1, dtype=np.int64)
        slots = self.rng.integers(0, global_counts)
        replace = slots < self.sample_limit
        if np.any(replace):
            self.sample[slots[replace]] = x32[start:][replace]
        self.count += remaining

    def finish(self):
        data = np.concatenate(self.chunks, axis=0) if self.sample_limit <= 0 else self.sample[:self.filled]
        if data.size == 0:
            raise ValueError("No quantile samples accumulated")
        return np.quantile(data, 0.01, axis=0), np.quantile(data, 0.99, axis=0)


def pad_feature(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[1] == dim:
        return x
    if x.shape[1] > dim:
        return x[:, :dim]
    out = np.zeros((x.shape[0], dim), dtype=x.dtype)
    out[:, :x.shape[1]] = x
    return out


def as_list(x: np.ndarray) -> list[float]:
    return [float(v) if math.isfinite(float(v)) else 0.0 for v in x]


def feature_stats(m: OnlineMoments, q: QuantileStore) -> dict:
    mean, std, mn, mx = m.finish()
    q01, q99 = q.finish()
    return {"mean": as_list(mean), "std": as_list(std), "min": as_list(mn),
            "max": as_list(mx), "q01": as_list(q01), "q99": as_list(q99)}


def load_episode_ranges(root: Path) -> list[tuple[int, int]]:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    paths += sorted((root / "meta" / "episodes").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode meta under {root / 'meta' / 'episodes'}")
    ranges: list[tuple[int, int]] = []
    for path in paths:
        df = pd.read_parquet(path, columns=["dataset_from_index", "dataset_to_index"])
        for a, b in df.itertuples(index=False):
            ranges.append((int(a), int(b)))
    ranges.sort()
    return ranges


def check_delta_alignment(root: Path, mask: np.ndarray, strict: bool) -> None:
    """Warn (or raise) when action/state are not aligned on delta-mask dims.

    The delta `action[d] - state[d]` is only meaningful when, for every mask=True dim d,
    action[d] and state[d] are the same physical quantity. This is an implicit assumption
    of both this script and the training transform; verify it from meta/info.json names.
    """
    info_path = None
    for name in ("info.json", "info.yaml", "info.yml"):
        cand = root / "meta" / name
        if cand.is_file():
            info_path = cand
            break
    if info_path is None or info_path.suffix != ".json":
        print(f"[align] WARNING: no meta/info.json under {root}; skipping action/state alignment check.")
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    feats = info.get("features", {})
    state_f = feats.get("observation.state", {})
    action_f = feats.get("action", {})
    state_names = list(state_f.get("names") or [])
    action_names = list(action_f.get("names") or [])
    state_dim = int(state_f.get("shape", [len(state_names)])[0]) if state_f.get("shape") else len(state_names)
    action_dim = int(action_f.get("shape", [len(action_names)])[0]) if action_f.get("shape") else len(action_names)

    print(f"[align] action_dim={action_dim} state_dim={state_dim} "
          f"delta_mask_true_dims={[int(i) for i in np.nonzero(mask)[0]]}")
    if action_dim != state_dim:
        print(f"[align] NOTE: action_dim ({action_dim}) != state_dim ({state_dim}); "
              f"both are padded/truncated to model_dim and delta only applies on mask dims.")

    if not state_names or not action_names:
        print("[align] WARNING: meta/info.json has no feature 'names'; "
              "cannot verify semantic alignment on delta-mask dims. Make sure action[d] and "
              "state[d] are the same quantity for every delta-mask=True dim.")
        if strict:
            raise ValueError("--strict-align set but feature names are missing for verification.")
        return

    problems: list[str] = []
    for d in np.nonzero(mask)[0]:
        d = int(d)
        a = action_names[d] if d < len(action_names) else None
        s = state_names[d] if d < len(state_names) else None
        if a is None or s is None or a != s:
            problems.append(f"  dim {d}: action='{a}' vs state='{s}'")

    if problems:
        msg = ("action/state are MISALIGNED on these delta-mask dims (delta would be "
               "computed between different quantities):\n" + "\n".join(problems))
        if strict:
            raise ValueError("[align] " + msg)
        print("[align] WARNING: " + msg)
        print("[align] Update DEFAULT_DELTA_MASK or the data so masked dims line up, "
              "or pass --strict-align to make this fatal.")
    else:
        print("[align] OK: action and state names match on all delta-mask dims.")


def data_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("*/*.parquet"))
    paths += sorted((root / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet under {root / 'data'}")
    return paths


def stream_episodes(
    root: Path,
    ranges: list[tuple[int, int]],
    dtype: np.dtype,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (ep_start, ep_end, states, actions) per episode while reading files lazily.

    Keeps only a small carry-over buffer covering the in-progress episode, so peak
    memory is about one parquet file plus the largest episode. Verifies that the global
    LeRobot index column is contiguous (required to build action horizons safely).
    """
    paths = data_parquet_paths(root)
    state = {"pi": 0, "idx0": None, "states": None, "actions": None, "end": None}

    def read_next_file() -> None:
        df = pd.read_parquet(paths[state["pi"]], columns=["index", "observation.state", "action"])
        state["pi"] += 1
        if df.empty:
            return
        idx = df["index"].to_numpy(dtype=np.int64)
        st = np.stack(df["observation.state"].to_numpy()).astype(dtype, copy=False)
        ac = np.stack(df["action"].to_numpy()).astype(dtype, copy=False)
        order = np.argsort(idx, kind="stable")
        idx, st, ac = idx[order], st[order], ac[order]
        if not np.array_equal(idx, idx[0] + np.arange(idx.shape[0], dtype=np.int64)):
            raise ValueError(f"Non-contiguous index within {paths[state['pi'] - 1]}")
        if state["states"] is None:
            state["idx0"] = int(idx[0])
            state["states"] = st
            state["actions"] = ac
            state["end"] = int(idx[0]) + idx.shape[0]
        else:
            if int(idx[0]) != state["end"]:
                raise ValueError(f"Gap between files: expected index {state['end']}, got {int(idx[0])}")
            state["states"] = np.concatenate([state["states"], st], axis=0)
            state["actions"] = np.concatenate([state["actions"], ac], axis=0)
            state["end"] += idx.shape[0]

    for ep_start, ep_end in ranges:
        if ep_end <= ep_start:
            continue
        while state["end"] is None or state["end"] < ep_end:
            if state["pi"] >= len(paths):
                raise ValueError(
                    f"Ran out of data files before covering episode [{ep_start}, {ep_end})"
                )
            read_next_file()
        # drop buffered rows before this episode start to release memory
        if ep_start > state["idx0"]:
            off = ep_start - state["idx0"]
            state["states"] = state["states"][off:].copy()
            state["actions"] = state["actions"][off:].copy()
            state["idx0"] = ep_start
        ls = ep_start - state["idx0"]
        le = ep_end - state["idx0"]
        yield ep_start, ep_end, state["states"][ls:le], state["actions"][ls:le]


def process_root(
    root: Path,
    *,
    dim: int,
    offsets: np.ndarray,
    mask: np.ndarray,
    dtype: np.dtype,
    s_mom: OnlineMoments,
    a_mom: OnlineMoments,
    s_q: QuantileStore,
    a_q: QuantileStore,
    desc: str,
) -> tuple[int, int, int]:
    """Stream one dataset root into the shared stat accumulators. Returns counts."""
    ranges = load_episode_ranges(root)
    n_states = 0
    n_actions = 0
    for _ep_start, _ep_end, state_ep, action_ep in tqdm(
        stream_episodes(root, ranges, dtype),
        total=len(ranges),
        desc=desc,
        unit="ep",
    ):
        n_ep = state_ep.shape[0]
        if n_ep <= 0:
            continue
        starts = np.arange(n_ep, dtype=np.int64)
        # clamp action horizon within the episode (repeat last frame), matching dataloader
        query = np.minimum(starts[:, None] + offsets[None, :], n_ep - 1)
        state = pad_feature(state_ep, dim)
        action_win = action_ep[query]  # (n_starts, horizon, raw_dim)
        n, h, _ = action_win.shape
        action_pad = pad_feature(action_win.reshape(n * h, action_win.shape[-1]), dim).reshape(n, h, dim)
        delta = action_pad
        if mask.any():
            delta[:, :, mask] -= state[:, None, mask]
        delta = delta.reshape(n * h, dim)

        s_mom.update(state)
        a_mom.update(delta)
        s_q.update(state)
        a_q.update(delta)
        n_states += int(state.shape[0])
        n_actions += int(delta.shape[0])
    return len(ranges), n_states, n_actions


def main() -> None:
    args = parse_args()
    roots = [Path(r) for r in args.data_root]
    dim, horizon = args.model_dim, args.action_horizon
    dtype = np.float64 if args.float64 else np.float32

    mask14 = np.asarray(DEFAULT_DELTA_MASK, dtype=bool)
    mask = np.zeros(dim, dtype=bool)
    mask[:min(dim, len(mask14))] = mask14[:dim]
    offsets = np.arange(horizon, dtype=np.int64)

    s_mom, a_mom = OnlineMoments(dim), OnlineMoments(dim)
    s_q = QuantileStore(dim, args.quantile_sample_limit, seed=13)
    a_q = QuantileStore(dim, args.quantile_sample_limit, seed=17)

    n_states = 0
    n_actions = 0
    n_episodes = 0
    roots_meta = []
    for root in roots:
        check_delta_alignment(root, mask, strict=args.strict_align)
        eps, r_states, r_actions = process_root(
            root, dim=dim, offsets=offsets, mask=mask, dtype=dtype,
            s_mom=s_mom, a_mom=a_mom, s_q=s_q, a_q=a_q,
            desc=f"[{args.task_name}] {root.name}",
        )
        n_episodes += eps
        n_states += r_states
        n_actions += r_actions
        roots_meta.append({
            "root": str(root),
            "num_episodes": eps,
            "num_state_samples": r_states,
            "num_action_samples": r_actions,
        })
        print(f"[{args.task_name}] {root}: episodes={eps} "
              f"state_samples={r_states} action_samples={r_actions}")

    result = {
        "norm_stats": {
            "observation.state": feature_stats(s_mom, s_q),
            "action": feature_stats(a_mom, a_q),
        }
    }
    if args.include_metadata:
        result["metadata"] = {
            "task_name": args.task_name,
            "data_roots": [str(r) for r in roots],
            "model_dim": dim,
            "action_horizon": horizon,
            "action_representation": "delta_for_true_delta_mask_dims_else_absolute_action",
            "boundary": "clamp_within_episode_repeat_last_frame (matches FastLeRobotDataset)",
            "delta_mask": list(DEFAULT_DELTA_MASK),
            "num_episodes": n_episodes,
            "num_state_samples": n_states,
            "num_action_samples": n_actions,
            "quantile_sample_limit": int(args.quantile_sample_limit),
            "roots": roots_meta,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"roots={len(roots)} episodes={n_episodes} "
          f"state_samples={n_states} action_samples={n_actions}")


if __name__ == "__main__":
    main()