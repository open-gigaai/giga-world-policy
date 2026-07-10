"""
Add episode-level T5 embedding cache to an existing LeRobot dataset.

What it does
------------
- For each episode, write a cached embedding file under
  `{dataset_root}/{t5_folder_name}/episode_XXXXXX.pt`.
- Supports legacy `meta/episodes.jsonl` and LeRobot v3-style
  `meta/episodes/*/*.parquet` + `meta/tasks.parquet`.
- If `meta/episodes.jsonl` exists, also writes `t5_embedding_path` pointers back
  to that file atomically. Parquet metadata is left unchanged.

Why
---
This script upgrades an existing LeRobot dataset by adding an episode-level T5 embedding
cache, so training/inference can run without on-the-fly text encoding. Newer
LeRobot v3/parquet datasets keep the cache as a sidecar directory and are not
rewritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_WAN_PATH = ""
DEFAULT_WAN_CODE_PATH = None
QUALITY_TASK_LABELS = {"qualified", "unqualified", "success", "failure"}


def _import_torch():
    import torch

    return torch


def _load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonlines_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _resolve_dataset_root(repo_id: Optional[str], root: Optional[str]) -> Path:
    if root is not None:
        root_path = Path(root).expanduser().resolve()
        if (root_path / "meta").is_dir():
            return root_path
    if repo_id is None:
        raise ValueError("Either --root or --repo_id must be provided.")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return Path(meta.root)


def _load_parquet_rows(paths: Iterable[Path], columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    paths = list(paths)
    if not paths:
        return []
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        rows = []
        for path in paths:
            try:
                frame = pd.read_parquet(path, columns=columns)
            except TypeError:
                frame = pd.read_parquet(path)
                if columns is not None:
                    present = [c for c in columns if c in frame.columns]
                    frame = frame[present]
            if not isinstance(frame.index, pd.RangeIndex) or frame.index.name is not None:
                frame = frame.reset_index()
                index_col = frame.index.name or "index"
                if index_col in frame.columns and "__index__" not in frame.columns:
                    frame = frame.rename(columns={index_col: "__index__"})
            rows.extend(frame.to_dict(orient="records"))
        return rows

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Reading LeRobot v3 parquet metadata requires pandas or pyarrow.") from error

    rows = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        rows.extend(table.to_pylist())
    return rows


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _normalize_tasks(value: Any, task_lookup: Dict[int, str]) -> List[str]:
    value = _scalar(value)
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            item = _scalar(item)
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif isinstance(item, bytes):
                text = item.decode("utf-8", errors="ignore").strip()
                if text:
                    out.append(text)
            else:
                try:
                    out.append(task_lookup[int(item)])
                except Exception:
                    text = str(item).strip()
                    if text:
                        out.append(text)
        return out
    try:
        return [task_lookup[int(value)]]
    except Exception:
        text = str(value).strip()
        return [text] if text else []


def _load_tasks(dataset_root: Path) -> Dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    rows = _load_parquet_rows([tasks_path])
    task_lookup = {}
    for row in rows:
        if "task_index" in row:
            key = int(_scalar(row["task_index"]))
        elif "index" in row:
            key = int(_scalar(row["index"]))
        else:
            key = len(task_lookup)
        name = row.get("task")
        if name is None:
            name = row.get("name")
        if name is None:
            name = row.get("__index__")
        if name is None:
            continue
        task_lookup[key] = str(_scalar(name))
    return task_lookup


def _load_episode_rows(dataset_root: Path) -> Tuple[List[Dict[str, Any]], Optional[Path], bool]:
    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        return _load_jsonlines(episodes_jsonl), episodes_jsonl, True

    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_paths = sorted(episodes_dir.glob("*/*.parquet")) + sorted(episodes_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata found under {dataset_root / 'meta'}")

    rows = _load_parquet_rows(parquet_paths)
    task_lookup = _load_tasks(dataset_root)
    normalized = []
    for row in rows:
        if "episode_index" not in row:
            raise KeyError(f"episode_index missing in LeRobot episode metadata under {episodes_dir}")
        ep = dict(row)
        if "tasks" not in ep:
            task_value = ep.get("task_index", ep.get("task", None))
            ep["tasks"] = _normalize_tasks(task_value, task_lookup)
        normalized.append(ep)
    normalized.sort(key=lambda item: int(_scalar(item["episode_index"])))
    return normalized, None, False


def _candidate_wan_code_paths(wan_code_path: Optional[str]) -> List[Path]:
    project_root = Path(__file__).resolve().parents[1]
    paths = [
        wan_code_path,
        os.environ.get("WAN_CODE_PATH"),
        os.environ.get("WAN_REPO_PATH"),
        DEFAULT_WAN_CODE_PATH,
        str(project_root / "third_party"),
    ]
    return [Path(p).expanduser().resolve() for p in paths if p]


def _ensure_wan_code_on_path(wan_code_path: Optional[str]) -> None:
    for path in _candidate_wan_code_paths(wan_code_path):
        if (path / "wan" / "modules" / "t5.py").is_file():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        if (path / "modules" / "t5.py").is_file() and path.name == "wan":
            parent_str = str(path.parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return
    raise FileNotFoundError(
        "Cannot find WAN python package. Pass --wan_code_path or set WAN_CODE_PATH "
        "to a directory that contains wan/modules/t5.py."
    )


def _load_wan_t5_encoder_class(wan_code_path: Optional[str]):
    for path in _candidate_wan_code_paths(wan_code_path):
        if (path / "wan" / "modules" / "t5.py").is_file():
            modules_dir = path / "wan" / "modules"
            package_name = "_wam_wan_t5"
            modules_package_name = f"{package_name}.modules"
            sys.modules.setdefault(package_name, types.ModuleType(package_name))
            modules_package = types.ModuleType(modules_package_name)
            modules_package.__path__ = [str(modules_dir)]
            sys.modules[modules_package_name] = modules_package
            for name in ("tokenizers", "t5"):
                module_name = f"{modules_package_name}.{name}"
                spec = importlib.util.spec_from_file_location(module_name, modules_dir / f"{name}.py")
                if spec is None or spec.loader is None:
                    raise ImportError(f"Failed to load WAN module {name} from {modules_dir}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            return sys.modules[f"{modules_package_name}.t5"].T5EncoderModel
    _ensure_wan_code_on_path(wan_code_path)
    from wan.modules.t5 import T5EncoderModel

    return T5EncoderModel


def _init_wan_t5_encoder(
    wan_path: str,
    device: str,
    text_len: int = 512,
    wan_code_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    torch = _import_torch()
    T5EncoderModel = _load_wan_t5_encoder_class(wan_code_path)

    ckpt = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "google/umt5-xxl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"T5 checkpoint not found: {ckpt}")
    if not os.path.exists(tok):
        raise FileNotFoundError(f"T5 tokenizer dir not found: {tok}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def _encode_t5(encoder: Any, instruction: str, device: str) -> torch.Tensor:
    torch = _import_torch()
    with torch.no_grad():
        out = encoder([instruction], device)
    if isinstance(out, list):
        emb = out[0]
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        raise ValueError(f"Unexpected T5 encoder output type: {type(out)}")

    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()


def _is_quality_label(text: str) -> bool:
    return text.strip().lower() in QUALITY_TASK_LABELS


def _select_instruction(tasks: List[str], mode: str) -> str:
    cleaned = [task for task in tasks if task and not _is_quality_label(task)]
    if not cleaned:
        return ""
    if mode == "first":
        return cleaned[0]
    if mode == "shortest":
        return min(cleaned, key=len)
    if mode == "longest":
        return max(cleaned, key=len)
    raise ValueError(f"Unsupported task_mode={mode!r}")


def _episode_instruction_from_meta(ep_row: Dict[str, Any], mode: str = "first") -> str:
    tasks = _normalize_tasks(ep_row.get("tasks", None), {})
    if tasks:
        return _select_instruction(tasks, mode=mode)
    task = ep_row.get("task", "")
    if isinstance(task, str):
        return task.strip()
    return str(task).strip()


def _iter_roots(root: Optional[str], root_list: Optional[str]) -> List[str]:
    roots: List[str] = []
    if root:
        roots.append(root)
    if root_list:
        with open(root_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    roots.append(line)
    if not roots:
        raise ValueError("No dataset root provided. Use --root or --root_list.")
    return roots


def _iter_root_specs(repo_id: Optional[str], root: Optional[str], root_list: Optional[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    if root or root_list:
        return [(repo_id, one_root) for one_root in _iter_roots(root, root_list)]
    if repo_id is None:
        raise ValueError("No dataset provided. Use --repo_id, --root, or --root_list.")
    return [(repo_id, None)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add WAN T5 embedding cache to a LeRobot dataset (episode-level pt + episodes.jsonl pointer)"
    )
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot dataset repo_id (identifier)")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root (contains meta/data/videos). If omitted, LeRobot default cache is used.",
    )
    parser.add_argument("--root_list", type=str, default=None, help="Text file with one local dataset root per line.")
    parser.add_argument(
        "--wan_path",
        type=str,
        default=None,
        help="Path that contains models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl.",
    )
    parser.add_argument(
        "--wan_code_path",
        type=str,
        default=None,
        help="Path that contains the WAN python package (wan/modules/t5.py). Default: env or project third_party.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--text_len", type=int, default=512, help="T5 text_len")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding", help="Cache folder name under dataset root")
    parser.add_argument(
        "--task_mode",
        choices=("first", "shortest", "longest"),
        default="first",
        help="How to choose one instruction when an episode has multiple LeRobot v3 tasks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files and meta pointers")
    parser.add_argument("--max_episodes", type=int, default=0, help="Process at most N episodes (0 = all)")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without loading WAN or writing files.")
    args = parser.parse_args()

    torch = None
    if args.device:
        device = args.device
    elif args.dry_run:
        device = "cpu"
    else:
        torch = _import_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.dry_run and torch is None:
        torch = _import_torch()
    wan_path = args.wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT") or DEFAULT_WAN_PATH
    if not wan_path and not args.dry_run:
        raise ValueError("WAN path not provided. Use --wan_path or set WAN_PATH/WAN_ROOT.")

    encoder = None
    total_updated = 0
    total_skipped = 0

    for repo_id, root in _iter_root_specs(args.repo_id, args.root, args.root_list):
        dataset_root = _resolve_dataset_root(repo_id, root)
        out_dir = dataset_root / args.t5_folder_name
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        episodes, episodes_path, can_write_pointers = _load_episode_rows(dataset_root)
        if args.max_episodes and args.max_episodes > 0:
            episodes = episodes[: args.max_episodes]

        updated = 0
        skipped = 0
        would_update = 0
        examples = []

        for ep in episodes:
            ep_idx = int(_scalar(ep["episode_index"]))
            rel = f"{args.t5_folder_name}/episode_{ep_idx:06d}.pt"
            abs_pt = dataset_root / rel

            has_ptr = ("t5_embedding_path" in ep) and isinstance(ep.get("t5_embedding_path"), str)
            ptr_ok = has_ptr and (dataset_root / str(ep["t5_embedding_path"])).exists()

            if not args.overwrite:
                if ptr_ok or abs_pt.exists():
                    if can_write_pointers and not has_ptr and abs_pt.exists():
                        ep["t5_embedding_path"] = rel
                        updated += 1
                    else:
                        skipped += 1
                    continue

            instr = _episode_instruction_from_meta(ep, mode=args.task_mode)
            if len(examples) < 3:
                examples.append((ep_idx, instr))
            if args.dry_run:
                would_update += 1
                continue

            if encoder is None:
                print(f"Loading WAN T5 encoder from {wan_path} on {device} ...")
                encoder = _init_wan_t5_encoder(
                    wan_path=wan_path,
                    device=device,
                    text_len=int(args.text_len),
                    wan_code_path=args.wan_code_path,
                )

            emb = _encode_t5(encoder, instr, device=device)
            torch.save(emb, abs_pt)
            ep["t5_embedding_path"] = rel
            updated += 1

            if updated % 10 == 0:
                print(f"Processed {updated} episodes in {dataset_root} (latest: {ep_idx})")

        if args.dry_run:
            print(f"[dry_run] root={dataset_root}")
            print(f"[dry_run] episodes={len(episodes)}, would_update={would_update}, skipped={skipped}")
            for ep_idx, instr in examples:
                print(f"[dry_run] example episode_{ep_idx:06d}: {instr}")
            total_updated += would_update
            total_skipped += skipped
            continue

        if can_write_pointers and episodes_path is not None:
            if args.max_episodes and args.max_episodes > 0:
                full = _load_jsonlines(episodes_path)
                patch_map = {int(_scalar(ep["episode_index"])): ep for ep in episodes}
                for i, ep in enumerate(full):
                    ep_idx = int(_scalar(ep["episode_index"]))
                    if ep_idx in patch_map:
                        full[i] = patch_map[ep_idx]
                _write_jsonlines_atomic(episodes_path, full)
            else:
                _write_jsonlines_atomic(episodes_path, episodes)

        print(f"Done root={dataset_root}. updated={updated}, skipped={skipped}.")
        total_updated += updated
        total_skipped += skipped

    print(f"Done all roots. updated={total_updated}, skipped={total_skipped}.")


if __name__ == "__main__":
    main()
