import json
import logging
import os
import shutil
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Callable

import av
import datasets
import datasets.config as hf_datasets_config
import lerobot.datasets.lerobot_dataset as _lerobot_dataset_module
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision
from filelock import FileLock, Timeout
from datasets.info import DatasetInfo
from datasets.naming import filenames_for_dataset_split
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _OriginalLeRobotDatasetMetadata
from lerobot.datasets.utils import (
    embed_images,
    validate_episode_buffer,
    validate_frame,
)
from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames, get_safe_default_codec
from PIL import Image
from typing_extensions import override

from .base_dataset import BaseDataset, _get_data_worker_context
from .dataset import register_dataset


TemporalOffsetSpec = int | list[int] | dict[str, int | list[int]]
VideoKeySpec = str | Sequence[str]


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning(f'Invalid integer value for {name}={value!r}; using {default}.')
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


class _LeRobotVideoNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True

        if not record.pathname.replace('\\', '/').endswith('/lerobot/datasets/video_utils.py'):
            return True

        message = record.getMessage()
        return not (
            message.startswith('Using video codec:')
            or message.startswith('Auto-selected video codec:')
            or message.startswith('No hardware encoder available, falling back to software encoder')
        )


_LEROBOT_VIDEO_NOISE_FILTER_MARKER = '_giga_lerobot_video_noise_filter'


def _install_lerobot_log_filters() -> None:
    if not _get_env_bool('GIGA_LEROBOT_FILTER_NOISY_LOGS', True):
        return

    warnings.filterwarnings(
        'ignore',
        message=r'The video decoding and encoding capabilities of torchvision are deprecated.*',
        category=UserWarning,
        module=r'torchvision\.io\._video_deprecation_warning',
    )

    root_logger = logging.getLogger()
    if any(getattr(log_filter, _LEROBOT_VIDEO_NOISE_FILTER_MARKER, False) for log_filter in root_logger.filters):
        return

    log_filter = _LeRobotVideoNoiseFilter()
    setattr(log_filter, _LEROBOT_VIDEO_NOISE_FILTER_MARKER, True)
    root_logger.addFilter(log_filter)


_install_lerobot_log_filters()


def _is_hf_parquet_cache_read_error(error: BaseException) -> bool:
    visited: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))

        message = str(current)
        if (
            '.incomplete' in message
            and '-of-NNNNN.arrow' in message
            and ('No such file or directory' in message or 'Errno 2' in message)
        ):
            return True
        if 'Expected to be able to read' in message and 'for message body, got' in message:
            return True
        if 'Tried reading schema message, was null or length 0' in message:
            return True

        cause = getattr(current, '__cause__', None)
        context = getattr(current, '__context__', None)
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return False


def _has_complete_hf_arrow_cache(cache_path: Path) -> bool:
    info_path = cache_path / hf_datasets_config.DATASET_INFO_FILENAME
    if not info_path.is_file():
        return False

    try:
        info = DatasetInfo.from_directory(str(cache_path))
    except Exception as error:
        logging.warning(f'Failed to inspect HF parquet cache metadata {info_path}: {error}')
        return False

    if info.splits is None:
        return False

    for split_info in info.splits.values():
        filenames = filenames_for_dataset_split(
            str(cache_path),
            dataset_name='parquet',
            split=split_info.name,
            filetype_suffix='arrow',
            shard_lengths=split_info.shard_lengths,
        )
        if any(not Path(filename).is_file() for filename in filenames):
            return False
    return True


def _clear_hf_parquet_builder_cache(builder_cache_dir: str | None) -> bool:
    if builder_cache_dir is None:
        return False

    cache_path = Path(builder_cache_dir)
    lock_path = f'{builder_cache_dir}_builder.lock'
    timeout_s = max(1, _get_env_int('GIGA_LEROBOT_HF_CACHE_REPAIR_LOCK_TIMEOUT_S', 600))
    try:
        with FileLock(lock_path, timeout=timeout_s):
            if _has_complete_hf_arrow_cache(cache_path):
                return False

            cleared = False
            for path in (cache_path, Path(f'{builder_cache_dir}.incomplete')):
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                        cleared = True
                    else:
                        existed = path.exists()
                        path.unlink(missing_ok=True)
                        cleared = cleared or existed
                except OSError as error:
                    logging.warning(f'Failed to clear HF parquet cache path {path}: {error}')
            return cleared
    except Timeout:
        logging.warning(
            f'Timed out after {timeout_s}s waiting to repair HF parquet cache {builder_cache_dir}; '
            'another process may still be building it.'
        )
        return False


def _format_lerobot_context(
    root: str | Path | None,
    *,
    repo_id: str | None = None,
    episodes: Sequence[int] | None = None,
    cache_dir: str | None = None,
    builder_cache_dir: str | None = None,
    parquet_count: int | None = None,
) -> str:
    parts = [f'root={root!s}']
    if repo_id is not None:
        parts.append(f'repo_id={repo_id}')
    if episodes is not None:
        preview = list(episodes[:10]) if not isinstance(episodes, list) else episodes[:10]
        suffix = '' if len(episodes) <= 10 else f'... total={len(episodes)}'
        parts.append(f'episodes={preview}{suffix}')
    if cache_dir is not None:
        parts.append(f'cache_dir={cache_dir}')
    if builder_cache_dir is not None:
        parts.append(f'builder_cache_dir={builder_cache_dir}')
    if parquet_count is not None:
        parts.append(f'parquet_count={parquet_count}')
    parts.append(_get_data_worker_context())
    return ', '.join(parts)


def _load_lerobot_episodes_no_hf_cache(root: Path) -> datasets.Dataset:
    from lerobot.datasets.utils import EPISODES_DIR

    paths = sorted((root / EPISODES_DIR).glob('*/*.parquet'))
    if len(paths) == 0:
        raise FileNotFoundError(f'Provided directory does not contain any parquet file: {root / EPISODES_DIR}')

    tables = []
    for path in paths:
        table = pq.read_table(path)
        columns = [name for name in table.column_names if not name.startswith('stats/')]
        tables.append(table.select(columns))

    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options='default')
    return datasets.Dataset.from_dict(table.to_pydict())


def _get_lerobot_total_frames_from_info(root: Path) -> int:
    with (root / 'meta' / 'info.json').open('r') as f:
        return int(json.load(f)['total_frames'])


def _get_lerobot_episode_lengths_no_hf_cache(root: Path, episodes: Sequence[int]) -> int:
    from lerobot.datasets.utils import EPISODES_DIR

    paths = sorted((root / EPISODES_DIR).glob('*/*.parquet'))
    if len(paths) == 0:
        raise FileNotFoundError(f'Provided directory does not contain any parquet file: {root / EPISODES_DIR}')

    columns = None
    episode_lengths: list[int] = []
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        if 'length' in schema_names:
            columns = ['length']
        elif {'dataset_from_index', 'dataset_to_index'}.issubset(schema_names):
            columns = ['dataset_from_index', 'dataset_to_index']
        else:
            raise KeyError(f'Missing episode length columns in {path}')

        table = pq.read_table(path, columns=columns)
        if 'length' in table.column_names:
            episode_lengths.extend(int(value) for value in table['length'].to_pylist())
        else:
            starts = table['dataset_from_index'].to_pylist()
            ends = table['dataset_to_index'].to_pylist()
            episode_lengths.extend(int(end) - int(start) for start, end in zip(starts, ends, strict=False))

    return sum(episode_lengths[int(episode_idx)] for episode_idx in episodes)


class _LeRobotDatasetMetadata(_OriginalLeRobotDatasetMetadata):
    def load_metadata(self):
        from lerobot.datasets.utils import (
            check_version_compatibility,
            load_info,
            load_stats,
            load_subtasks,
            load_tasks,
        )

        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, _lerobot_dataset_module.CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.subtasks = load_subtasks(self.root)
        self.episodes = _load_lerobot_episodes_no_hf_cache(self.root)
        self.stats = load_stats(self.root)


_lerobot_dataset_module.LeRobotDatasetMetadata = _LeRobotDatasetMetadata


def _is_valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_delta_offsets(delta_name: str, delta_spec: TemporalOffsetSpec) -> list[int]:
    if _is_valid_int(delta_spec):
        if delta_spec < 0:
            raise ValueError(f"delta_info['{delta_name}'] must be non-negative when provided as an int.")
        return list(range(delta_spec))

    if isinstance(delta_spec, Sequence) and not isinstance(delta_spec, (str, bytes)):
        offsets: list[int] = []
        for offset in delta_spec:
            if not _is_valid_int(offset):
                raise TypeError(
                    f"delta_info['{delta_name}'] offsets must be integers, got {type(offset).__name__}."
                )
            offsets.append(offset)
        return offsets

    if not isinstance(delta_spec, Mapping):
        raise TypeError(
            f"delta_info['{delta_name}'] must be an int, a list[int], or a dict, got {type(delta_spec).__name__}."
        )

    has_offsets = 'offsets' in delta_spec
    has_window = any(key in delta_spec for key in ('start', 'stop', 'stride'))
    if has_offsets and has_window:
        raise ValueError(
            f"delta_info['{delta_name}'] cannot mix 'offsets' with 'start'/'stop'/'stride'."
        )

    if has_offsets:
        unknown_keys = set(delta_spec) - {'offsets'}
        if unknown_keys:
            raise ValueError(
                f"delta_info['{delta_name}'] has unsupported keys for offsets mode: {sorted(unknown_keys)}."
            )
        return _normalize_delta_offsets(delta_name, delta_spec['offsets'])

    unknown_keys = set(delta_spec) - {'start', 'stop', 'stride'}
    if unknown_keys:
        raise ValueError(
            f"delta_info['{delta_name}'] has unsupported keys for range mode: {sorted(unknown_keys)}."
        )
    if 'start' not in delta_spec or 'stop' not in delta_spec:
        raise ValueError(f"delta_info['{delta_name}'] range mode requires both 'start' and 'stop'.")

    start = delta_spec['start']
    stop = delta_spec['stop']
    stride = delta_spec.get('stride', 1)
    if not _is_valid_int(start) or not _is_valid_int(stop) or not _is_valid_int(stride):
        raise TypeError(
            f"delta_info['{delta_name}'] range mode expects integer 'start', 'stop', and 'stride'."
        )
    if stride <= 0:
        raise ValueError(f"delta_info['{delta_name}'] stride must be a positive integer.")
    return list(range(start, stop, stride))


def _build_delta_timestamps(
    delta_info: dict[str, TemporalOffsetSpec] | None,
    fps: int,
) -> dict[str, list[float]] | None:
    if delta_info is None:
        return None

    delta_timestamps: dict[str, list[float]] = {}
    for delta_name, delta_spec in delta_info.items():
        delta_timestamps[delta_name] = [offset / fps for offset in _normalize_delta_offsets(delta_name, delta_spec)]
    return delta_timestamps


def _normalize_video_key_spec(video_keys: VideoKeySpec | None, *, arg_name: str) -> tuple[str, ...] | None:
    if video_keys is None:
        return None
    if isinstance(video_keys, str):
        return (video_keys,)
    if not isinstance(video_keys, Iterable):
        raise TypeError(f'{arg_name} should be a string, a sequence of strings, or None.')

    normalized = []
    seen = set()
    for key in video_keys:
        if not isinstance(key, str):
            raise TypeError(f'{arg_name} should contain strings, got {type(key).__name__}.')
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    return tuple(normalized)


def _select_nearest_frame(
    previous_frame: tuple[float, torch.Tensor] | None,
    current_frame: tuple[float, torch.Tensor] | None,
    query_timestamp: float,
) -> tuple[float, torch.Tensor]:
    if previous_frame is None and current_frame is None:
        raise FrameTimestampError(f'No frame could be decoded for timestamp {query_timestamp}.')
    if previous_frame is None:
        return current_frame
    if current_frame is None:
        return previous_frame

    previous_distance = abs(previous_frame[0] - query_timestamp)
    current_distance = abs(current_frame[0] - query_timestamp)
    if previous_distance <= current_distance:
        return previous_frame
    return current_frame


def _raise_frame_tolerance_error(
    video_path: Path | str,
    timestamps: list[float],
    query_timestamp: float,
    matched_timestamp: float,
    tolerance_s: float,
) -> None:
    raise FrameTimestampError(
        f'Closest frame violates tolerance ({abs(matched_timestamp - query_timestamp)} >= {tolerance_s=}).'
        ' It means that the closest frame that can be loaded from the video is too far away in time.'
        ' This might be due to synchronization issues with timestamps during data collection.'
        ' To be safe, we advise to ignore this item during training.'
        f'\nqueried timestamps: {torch.tensor(timestamps)}'
        f'\nclosest timestamp: {matched_timestamp}'
        f'\nvideo: {video_path}'
        '\nbackend: pyav'
    )


def _decode_video_frames_pyav_streaming(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode timestamps with bounded CPU working set for the pyav backend.

    Unlike torchvision's default implementation, this keeps at most the previous
    and current decoded frames while scanning forward from the nearest keyframe.
    """
    if not timestamps:
        raise ValueError('timestamps must not be empty.')

    sorted_query_positions = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    sorted_timestamps = [timestamps[pos] for pos in sorted_query_positions]

    torchvision.set_video_backend('pyav')
    reader = torchvision.io.VideoReader(str(video_path), 'video')
    reader.seek(sorted_timestamps[0], keyframes_only=True)

    selected_frames: list[torch.Tensor | None] = [None] * len(timestamps)
    selected_timestamps: list[float | None] = [None] * len(timestamps)
    previous_frame: tuple[float, torch.Tensor] | None = None
    query_cursor = 0

    try:
        for frame in reader:
            current_timestamp = float(frame['pts'])
            current_frame = (current_timestamp, frame['data'])

            if log_loaded_timestamps:
                logging.info(f'frame loaded at timestamp={current_timestamp:.4f}')

            while query_cursor < len(sorted_timestamps) and current_timestamp >= sorted_timestamps[query_cursor]:
                matched_timestamp, matched_frame = _select_nearest_frame(
                    previous_frame,
                    current_frame,
                    sorted_timestamps[query_cursor],
                )
                if abs(matched_timestamp - sorted_timestamps[query_cursor]) >= tolerance_s:
                    _raise_frame_tolerance_error(
                        video_path,
                        timestamps,
                        sorted_timestamps[query_cursor],
                        matched_timestamp,
                        tolerance_s,
                    )

                original_position = sorted_query_positions[query_cursor]
                selected_frames[original_position] = matched_frame
                selected_timestamps[original_position] = matched_timestamp
                query_cursor += 1

            previous_frame = current_frame
            if query_cursor == len(sorted_timestamps):
                break
    finally:
        container = getattr(reader, 'container', None)
        if container is not None:
            container.close()
        reader = None

    if previous_frame is None:
        raise FrameTimestampError(f'No frames could be decoded from video: {video_path}')

    while query_cursor < len(sorted_timestamps):
        matched_timestamp, matched_frame = previous_frame
        if abs(matched_timestamp - sorted_timestamps[query_cursor]) >= tolerance_s:
            _raise_frame_tolerance_error(
                video_path,
                timestamps,
                sorted_timestamps[query_cursor],
                matched_timestamp,
                tolerance_s,
            )

        original_position = sorted_query_positions[query_cursor]
        selected_frames[original_position] = matched_frame
        selected_timestamps[original_position] = matched_timestamp
        query_cursor += 1

    closest_frames = torch.stack([frame for frame in selected_frames if frame is not None])
    closest_frames = closest_frames.type(torch.float32) / 255

    if log_loaded_timestamps:
        logging.info(f'closest_timestamps={torch.tensor(selected_timestamps)}')

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f'Number of retrieved frames ({len(closest_frames)}) does not match '
            f'number of queried timestamps ({len(timestamps)})'
        )
    return closest_frames


@register_dataset
class LeRobotDataset(BaseDataset):
    """Wrapper around lerobot's dataset for unified interface with our
    BaseDataset.

    Supports optional ``delta_info`` to compute additional timestamp offsets and
    attaching dataset-level meta under ``meta_name`` in each returned sample.
    ``delta_info`` values can be legacy integers, explicit signed offset lists,
    or structured ``{'start', 'stop', 'stride'}`` windows.
    """

    def __init__(
        self,
        data_path: str,
        data_size: int | None = None,
        delta_info: dict[str, TemporalOffsetSpec] | None = None,
        meta_name: str | None = None,
        repack_transform: dict | None = None,
        decode_video_keys: VideoKeySpec | None = None,
        **kwargs,
    ) -> None:
        super(LeRobotDataset, self).__init__(data_path=data_path)
        self.data_size = data_size
        self.delta_info = delta_info
        self.meta_name = meta_name
        self.repack_transform = repack_transform
        self.decode_video_keys = _normalize_video_key_spec(decode_video_keys, arg_name='decode_video_keys')
        self.kwargs = kwargs
        self.dataset = None

    def _load_metadata(self) -> _LeRobotDatasetMetadata:
        repo_id = os.path.basename(os.path.normpath(self.data_path))
        return _LeRobotDatasetMetadata(repo_id, root=self.data_path)

    def _get_metadata_data_size(self) -> int:
        root = Path(self.data_path)
        episodes = self.kwargs.get('episodes')
        if episodes is None:
            return _get_lerobot_total_frames_from_info(root)
        return _get_lerobot_episode_lengths_no_hf_cache(root, episodes)

    @classmethod
    def load(cls, data_or_config: str | dict) -> 'LeRobotDataset':
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def open(self) -> None:
        if self.dataset is None:
            try:
                # Build metadata and optionally derive delta timestamps per stream
                dataset_meta = self._load_metadata()
                delta_timestamps = _build_delta_timestamps(self.delta_info, dataset_meta.fps)
                # Use faster subclass that avoids temporary disk writes during conversion
                self.dataset = FastLeRobotDataset(
                    dataset_meta.repo_id,
                    root=self.data_path,
                    delta_timestamps=delta_timestamps,
                    video_backend='pyav',
                    repack_transform=self.repack_transform,
                    decode_video_keys=self.decode_video_keys,
                    **self.kwargs,
                )
                if self.data_size is not None:
                    assert self.data_size == len(self.dataset)
                else:
                    self.data_size = len(self.dataset)
            except Exception as error:
                raise RuntimeError(
                    'Failed to open LeRobotDataset: '
                    f'{_format_lerobot_context(self.data_path, episodes=self.kwargs.get("episodes"))}, '
                    f'delta_keys={list(self.delta_info or {})}, kwargs_keys={list(self.kwargs)}'
                ) from error

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset = None
        super(LeRobotDataset, self).close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.data_size = self._get_metadata_data_size()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        data_dict = self.dataset[index]
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            # Attach global meta under user-provided key
            data_dict[self.meta_name] = self.dataset.meta
        return data_dict


class FastLeRobotDataset(_LeRobotDataset):
    """This class overrides the `LeRobotDataset`(lerobot version 0.4.4) class
    to accelerate the data conversion process.

    What it does is:
    - Doesn't store temporary image files to disk, instead, it's kept in memory until the whole episode is saved.
    - Only consider observation.state and action features to compute episode statistics.

    Beside, it's recommended to use video mode rather than image mode when converting large datasets. It's easy for data transfer and storage.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 2e-2,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
        skip_video_decoding: bool = False,
        repack_transform: dict | None = None,
        decode_video_keys: VideoKeySpec | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self.skip_video_decoding = skip_video_decoding
        self.repack_transform = repack_transform
        self.decode_video_keys = _normalize_video_key_spec(decode_video_keys, arg_name='decode_video_keys')
        self._decode_video_key_set: set[str] | None = None
        self._repack_mode = 'none'
        self._repack_pairs = ()
        self._repack_assignments = ()
        # Tracks (src_key, dst_key) pairs whose source has already been reported
        # as missing for this dataset instance, so we warn once per pair instead
        # of spamming the log on every __getitem__.
        self._repack_warned_missing: set[tuple[str, str]] = set()
        if repack_transform:
            has_nested = any(isinstance(value, dict) for value in repack_transform.values())
            if has_nested:
                # Structured mode: map source flat keys into target nested structure.
                self._repack_mode = 'structured'
                self._repack_assignments = self._build_repack_assignments(repack_transform)
            else:
                # Flat mode: support both src->dst and dst->src styles at runtime.
                self._repack_mode = 'flat'
                self._repack_pairs = self._build_repack_pairs(repack_transform)

    @override
    def load_hf_dataset(self) -> datasets.Dataset:
        """Load parquet data using the metadata-declared columns only.

        Some converted LeRobot shards contain extra parquet columns that are not
        present in ``meta/info.json``. HuggingFace Datasets raises a CastError
        when casting such files with an explicit features schema. Restricting the
        parquet read to the declared feature columns preserves LeRobot's
        metadata contract and ignores unused extra columns.
        """
        from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
        from datasets.io.parquet import ParquetDatasetReader
        from pyarrow import dataset as pa_ds

        features = get_hf_features_from_features(self.features)
        paths = sorted((self.root / 'data').glob('*/*.parquet'))
        if len(paths) == 0:
            raise FileNotFoundError(f'Provided directory does not contain any parquet file: {self.root / "data"}')

        filters = pa_ds.field('episode_index').isin(self.episodes) if self.episodes is not None else None
        cache_dir = os.environ.get('HF_DATASETS_CACHE') or os.path.expanduser('~/.cache/huggingface/datasets')
        parquet_paths = [str(path) for path in paths]
        builder_cache_dir: str | None = None

        def load_from_parquet() -> datasets.Dataset:
            nonlocal builder_cache_dir
            reader = ParquetDatasetReader(
                parquet_paths,
                filters=filters,
                features=features,
                columns=list(features),
                cache_dir=cache_dir,
            )
            builder_cache_dir = reader.builder.cache_dir
            return reader.read()

        def load_from_parquet_with_retry() -> datasets.Dataset:
            attempts = max(1, _get_env_int('GIGA_LEROBOT_HF_CACHE_LOAD_ATTEMPTS', 10))
            for attempt in range(attempts):
                try:
                    return load_from_parquet()
                except Exception as error:
                    is_recoverable_cache_error = _is_hf_parquet_cache_read_error(error)
                    if not is_recoverable_cache_error:
                        raise
                    if attempt + 1 == attempts:
                        raise RuntimeError(
                            'HF parquet cache remained missing or truncated after '
                            f'{attempts} attempts: '
                            f'{_format_lerobot_context(self.root, repo_id=self.repo_id, episodes=self.episodes, cache_dir=cache_dir, builder_cache_dir=builder_cache_dir, parquet_count=len(parquet_paths))}'
                        ) from error
                    cleared_cache = _clear_hf_parquet_builder_cache(builder_cache_dir)
                    delay_s = min(2.0, 0.2 * (2**attempt))
                    logging.warning(
                        'HF parquet cache is missing or truncated; '
                        f'{"cleared" if cleared_cache else "kept"} {builder_cache_dir} and retrying in {delay_s:.1f}s '
                        f'({attempt + 1}/{attempts}). '
                        f'{_format_lerobot_context(self.root, repo_id=self.repo_id, episodes=self.episodes, cache_dir=cache_dir, builder_cache_dir=builder_cache_dir, parquet_count=len(parquet_paths))}'
                    )
                    time.sleep(delay_s)
            raise RuntimeError('Unreachable HF parquet cache retry state.')

        if _get_env_bool('GIGA_LEROBOT_LOCK_HF_CACHE', True):
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            lock_path = cache_path / 'giga_lerobot_from_parquet.lock'
            with FileLock(str(lock_path)):
                hf_dataset = load_from_parquet_with_retry()
        else:
            hf_dataset = load_from_parquet_with_retry()

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @override
    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """This function only adds the frame to the episode_buffer and nothing
        is written to disk.

        To save those frames, the 'save_episode()' method then needs to be called.
        """
        # Convert torch tensors to numpy arrays for serialization/storage
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer['size']
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer['frame_index'].append(frame_index)
        self.episode_buffer['timestamp'].append(timestamp)
        self.episode_buffer['task'].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")

            self.episode_buffer[key].append(frame[key])

        self.episode_buffer['size'] += 1

    @override
    def save_episode(self, episode_data: dict | None = None) -> None:
        """This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # 'size' and 'task' are bookkeeping fields, omitted from parquet payload
        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length,), episode_index)

        # Register any new tasks encountered during this episode.
        self.meta.save_episode_tasks(episode_tasks)

        # Map natural-language task names to task indices
        episode_buffer['task_index'] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ['index', 'episode_index', 'task_index'] or ft['dtype'] in ['image', 'video']:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = _compute_episode_stats(episode_buffer)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_buffer, episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # Persist episode meta after encoding videos to include video metadata.
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, {})

        video_files = list(self.root.rglob('*.mp4'))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob('*.parquet'))
        assert len(parquet_files) == self.num_episodes

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    @override
    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split='train')
        ep_dataset = embed_images(ep_dataset)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    @override
    def encode_episode_videos(self, episode_buffer: dict, episode_index: int) -> dict:
        """Use ffmpeg to convert frames stored as png into mp4 videos.

        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue

            imgs = episode_buffer[key]

            _encode_video_frames(imgs, video_path, self.fps, overwrite=True)

        return video_paths

    @override
    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        ep = self.meta.episodes[ep_idx]
        item = {}
        video_backend = self.video_backend or get_safe_default_codec()
        query_timestamps = self._filter_query_timestamps(query_timestamps)
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f'videos/{vid_key}/from_timestamp']
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            if video_backend == 'pyav':
                frames = _decode_video_frames_pyav_streaming(video_path, shifted_query_ts, self.tolerance_s)
            else:
                frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    @override
    def __getitem__(self, idx: int) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item['episode_index'].item()
        abs_idx = item['index'].item()

        query_indices = None
        if self.delta_indices is not None:
            # lerobot>=0.4 expects absolute dataset index here.
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Optional: skip costly video decoding when only computing stats
        if not self.skip_video_decoding and len(self.meta.video_keys) > 0:
            current_ts = item['timestamp'].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                logging.warning(
                    f'Failed to decode video frames for episode {ep_idx} in timestamps: {query_timestamps}. Error: {e}. Falling back to zeros.'
                )
                video_frames = {}
                # Construct zero tensors matching expected shapes per key
                for vid_key, query_ts in self._filter_query_timestamps(query_timestamps).items():
                    num_queries = len(query_ts)
                    # Prefer shapes from metadata when available
                    ft_shape = self.meta.shapes.get(vid_key)

                    # Derive channel-first (C,H,W)
                    if isinstance(ft_shape, tuple) and len(ft_shape) == 3:
                        if ft_shape[0] in (1, 3, 4):  # likely CHW
                            c, h, w = ft_shape[0], ft_shape[1], ft_shape[2]
                        elif ft_shape[2] in (1, 3, 4):  # likely HWC
                            c, h, w = ft_shape[2], ft_shape[0], ft_shape[1]
                        else:
                            c, h, w = 3, ft_shape[0], ft_shape[1]
                    else:
                        # Conservative default
                        c, h, w = 3, 224, 224

                    if num_queries > 1:
                        zeros_shape = (num_queries, c, h, w)
                    else:
                        zeros_shape = (c, h, w)

                    video_frames[vid_key] = torch.zeros(zeros_shape, dtype=torch.float32)

            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item['task_index'].item()
        item['task'] = self.meta.tasks.iloc[task_idx].name
        if self._repack_mode != 'none':
            item = self._apply_repack_transform(item)
        return item

    def _filter_query_timestamps(self, query_timestamps: dict[str, list[float]]) -> dict[str, list[float]]:
        decode_video_key_set = self._get_decode_video_key_set()
        if decode_video_key_set is None:
            return query_timestamps
        return {key: value for key, value in query_timestamps.items() if key in decode_video_key_set}

    def _get_decode_video_key_set(self) -> set[str] | None:
        if self.decode_video_keys is None:
            return None
        if self._decode_video_key_set is None:
            available_keys = set(self.meta.video_keys)
            resolved_keys: set[str] = set()
            missing_keys = []
            for requested_key in self.decode_video_keys:
                candidates = self._candidate_source_keys_for_decode_key(requested_key)
                source_key = next((key for key in candidates if key in available_keys), None)
                if source_key is None:
                    missing_keys.append(requested_key)
                    continue
                resolved_keys.add(source_key)

            if missing_keys:
                print(
                    f'[decode_video_keys] dataset {self.root!s}: requested keys {missing_keys!r} '
                    f'not present in video keys {sorted(available_keys)!r}; skipping them.',
                    flush=True,
                )
            self._decode_video_key_set = resolved_keys
        return self._decode_video_key_set

    def _candidate_source_keys_for_decode_key(self, requested_key: str) -> tuple[str, ...]:
        candidates = [requested_key]
        if self._repack_mode == 'flat':
            for left_key, right_key in self._repack_pairs:
                if left_key == requested_key and right_key not in candidates:
                    candidates.append(right_key)
                if right_key == requested_key and left_key not in candidates:
                    candidates.append(left_key)
        elif self._repack_mode == 'structured':
            for dst_path, src_key in self._repack_assignments:
                dst_key = '.'.join(dst_path)
                if dst_key == requested_key and src_key not in candidates:
                    candidates.append(src_key)
                if len(dst_path) == 1 and dst_path[0] == requested_key and src_key not in candidates:
                    candidates.append(src_key)
        return tuple(candidates)

    @staticmethod
    def _build_repack_pairs(repack_transform: dict[str, str]) -> tuple[tuple[str, str], ...]:
        pairs = [(src_key, dst_key) for src_key, dst_key in repack_transform.items() if src_key != dst_key]
        explicit_src_keys = {src_key for src_key, _ in pairs}
        explicit_dst_keys = {dst_key for _, dst_key in pairs}

        for src_key, dst_key in tuple(pairs):
            if src_key.endswith('_is_pad') or dst_key.endswith('_is_pad'):
                continue

            src_pad_key = f'{src_key}_is_pad'
            dst_pad_key = f'{dst_key}_is_pad'
            if src_pad_key in explicit_src_keys or dst_pad_key in explicit_dst_keys:
                continue
            if src_pad_key == dst_pad_key:
                continue

            pairs.append((src_pad_key, dst_pad_key))

        return tuple(pairs)

    @staticmethod
    def _collect_repack_leaf_mappings(repack_transform: dict) -> tuple[tuple[tuple[str, ...], str], ...]:
        assignments: list[tuple[tuple[str, ...], str]] = []

        def _walk(node: dict, prefix: tuple[str, ...] = ()) -> None:
            for dst_key, mapping in node.items():
                dst_path = prefix + (dst_key,)
                if isinstance(mapping, dict):
                    _walk(mapping, dst_path)
                    continue
                if not isinstance(mapping, str):
                    raise TypeError(
                        f'Invalid repack_transform leaf type for key {dst_path}. '
                        f'Expected str or dict, got {type(mapping).__name__}.'
                    )
                assignments.append((dst_path, mapping))

        _walk(repack_transform)
        return tuple(assignments)

    @staticmethod
    def _build_repack_assignments(repack_transform: dict) -> tuple[tuple[tuple[str, ...], str], ...]:
        explicit_mappings = FastLeRobotDataset._collect_repack_leaf_mappings(repack_transform)
        explicit_src_keys = {src_key for _, src_key in explicit_mappings}
        explicit_dst_paths = {dst_path for dst_path, _ in explicit_mappings}
        assignments: list[tuple[tuple[str, ...], str]] = []

        for dst_path, src_key in explicit_mappings:
            # No-op mapping (e.g. {'action': 'action'}) can be skipped.
            if len(dst_path) != 1 or dst_path[0] != src_key:
                assignments.append((dst_path, src_key))

            if src_key.endswith('_is_pad') or dst_path[-1].endswith('_is_pad'):
                continue

            src_pad_key = f'{src_key}_is_pad'
            dst_pad_path = dst_path[:-1] + (f'{dst_path[-1]}_is_pad',)
            if src_pad_key in explicit_src_keys or dst_pad_path in explicit_dst_paths:
                continue
            if len(dst_pad_path) == 1 and dst_pad_path[0] == src_pad_key:
                continue

            assignments.append((dst_pad_path, src_pad_key))

        return tuple(assignments)

    def _warn_repack_miss(self, src_key: str, dst_key: str) -> None:
        """Log once per (src, dst) pair that the configured source key is absent.

        Dataset groups in giga-brain configs share a single repack_transform,
        but individual paths may legitimately lack some optional cameras. We
        skip those mappings silently for training, but emit a single warning
        per dataset instance + key pair so the path can be located.
        """
        if src_key.endswith('_is_pad') or dst_key.endswith('_is_pad'):
            # Auto-derived pad pairs follow their parent; warning on the parent
            # is enough — don't double-log.
            return
        pair = (src_key, dst_key)
        if pair in self._repack_warned_missing:
            return
        self._repack_warned_missing.add(pair)
        print(
            f'[repack_transform] dataset {self.root!s}: source key {src_key!r} '
            f'(target {dst_key!r}) not present in sample; skipping this mapping.',
            flush=True,
        )

    def _apply_repack_transform(self, item: dict) -> dict:
        # Rename/repack in place to avoid allocating a new dict per sample.
        if self._repack_mode == 'flat':
            for left_key, right_key in self._repack_pairs:
                left_exists = left_key in item
                right_exists = right_key in item

                if left_exists and not right_exists:
                    src_key, dst_key = left_key, right_key
                elif right_exists and not left_exists:
                    src_key, dst_key = right_key, left_key
                elif left_exists and right_exists:
                    # Ambiguous case: keep backward-compatible behavior (left -> right).
                    src_key, dst_key = left_key, right_key
                else:
                    self._warn_repack_miss(left_key, right_key)
                    continue

                item[dst_key] = item.pop(src_key)

            return item

        if self._repack_mode == 'structured':
            for dst_path, src_key in self._repack_assignments:
                if src_key not in item:
                    self._warn_repack_miss(src_key, '.'.join(dst_path))
                    continue

                value = item.pop(src_key)

                if len(dst_path) == 1:
                    item[dst_path[0]] = value
                    continue

                parent = item
                for path_key in dst_path[:-1]:
                    child = parent.get(path_key)
                    if not isinstance(child, dict):
                        child = {}
                        parent[path_key] = child
                    parent = child
                parent[dst_path[-1]] = value

        return item


def _encode_video_frames(
    imgs: list[np.ndarray],
    video_path: Path | str,
    fps: int,
    vcodec: str = 'libsvtav1',
    pix_fmt: str = 'yuv420p',
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """Encode a sequence of RGB frames into a video file using PyAV/ffmpeg.

    Args:
        imgs (list[np.ndarray]): List of frames as HxWx3 RGB numpy arrays. All frames
            are assumed to share the same spatial size.
        video_path (Path | str): Output path for the encoded video file.
        fps (int): Target frames per second for the output video.
        vcodec (str): Video codec passed to ffmpeg. Supported: {'h264', 'hevc', 'libsvtav1'}.
        pix_fmt (str): Pixel format for the encoder (e.g., 'yuv420p', 'yuv444p').
        g (int | None): GOP size (distance between keyframes). ``None`` keeps encoder default.
        crf (int | None): Constant Rate Factor controlling quality/bitrate (lower is higher quality).
            ``None`` keeps encoder default.
        fast_decode (int): Enable fast-decode tuning when non-zero (codec-dependent behavior).
        log_level (int | None): LibAV logging level (e.g., ``av.logging.ERROR``). ``None`` keeps default.
        overwrite (bool): If True, create parent directories as needed and allow overwriting.

    Raises:
        ValueError: If ``vcodec`` is unsupported.
        FileNotFoundError: If no frames are provided.
        OSError: If encoding appears to succeed but the output file is missing.

    Notes:
        More details about ffmpeg argument tuning can be found in `benchmark/video/README.md`.
    """
    # Check encoder availability
    if vcodec not in ['h264', 'hevc', 'libsvtav1']:
        raise ValueError(f'Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.')

    video_path = Path(video_path)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == 'libsvtav1' or vcodec == 'hevc') and pix_fmt == 'yuv444p':
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = 'yuv420p'

    # Define video output frame size (assuming all input frames are the same size)
    if len(imgs) == 0:
        raise FileNotFoundError('No images found.')
    dummy_image = Image.fromarray(imgs[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options['g'] = str(g)

    if crf is not None:
        video_options['crf'] = str(crf)

    if fast_decode:
        key = 'svtav1-params' if vcodec == 'libsvtav1' else 'tune'
        value = f'fast-decode={fast_decode}' if vcodec == 'libsvtav1' else 'fastdecode'
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger('libav').setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), 'w') as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in imgs:
            input_image = Image.fromarray(input_data).convert('RGB')
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f'Video encoding did not work. File not found: {video_path}.')


def _get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        'min': np.min(array, axis=axis, keepdims=keepdims),
        'max': np.max(array, axis=axis, keepdims=keepdims),
        'mean': np.mean(array, axis=axis, keepdims=keepdims),
        'std': np.std(array, axis=axis, keepdims=keepdims),
        'count': np.array([len(array)]),
    }


def _compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray]) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in ['observation.state', 'action']:
            continue

        ep_ft_array = data  # data is already a np.ndarray
        axes_to_reduce = 0  # compute stats over the first axis
        keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = _get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

    return ep_stats
