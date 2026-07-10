import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from ..datasets import ConcatDataset


class ListWeightedSampler(Sampler):
    """Sampler for ``ConcatDataset`` using explicit per-subdataset weights.

    Sampling weights represent sampling probabilities among child datasets.
    The sampled total size follows the concatenated dataset size (optionally
    padded to ``batch_size``). Sampling ratios can be enforced at the epoch
    level or within each ``batch_size`` block.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        sampling_weights: list[float] | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        ratio_mode: str = 'epoch',
        index_mode: str = 'array',
        epoch_size: int | None = None,
        chunk_size: int = 8192,
        shuffle_within_chunk: bool = False,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError('dataset should be a ConcatDataset')

        if sampling_weights is None:
            sampling_weights = getattr(dataset, 'sampling_weights', None)
        if sampling_weights is None:
            raise ValueError('sampling_weights is required, or dataset should provide sampling_weights')

        if len(sampling_weights) != len(dataset.datasets):
            raise ValueError('sampling_weights length should match number of child datasets')

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.ratio_mode = ratio_mode
        self.index_mode = index_mode
        self.epoch_size = epoch_size
        self.chunk_size = int(chunk_size)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.epoch = 0

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError('batch_size should be greater than 0')
        if self.ratio_mode not in ('epoch', 'batch'):
            raise ValueError("ratio_mode should be either 'epoch' or 'batch'")
        if self.ratio_mode == 'batch' and self.batch_size is None:
            raise ValueError("batch_size is required when ratio_mode is 'batch'")
        if self.index_mode not in ('array', 'chunk'):
            raise ValueError("index_mode should be either 'array' or 'chunk'")
        if self.index_mode == 'chunk' and self.ratio_mode != 'epoch':
            raise ValueError("index_mode='chunk' currently supports ratio_mode='epoch' only")
        if self.chunk_size <= 0:
            raise ValueError('chunk_size should be greater than 0')
        if self.epoch_size is not None and self.epoch_size <= 0:
            raise ValueError('epoch_size should be greater than 0')

        self.sampling_weights = [float(w) for w in sampling_weights]
        if any(w < 0 for w in self.sampling_weights):
            raise ValueError('sampling_weights should be non-negative')
        weight_sum = sum(self.sampling_weights)
        if weight_sum <= 0:
            raise ValueError('sum of sampling_weights should be greater than 0')

        self.sub_dataset_lengths = [len(d) for d in self.dataset.datasets]
        empty_dataset_indices = [i for i, length in enumerate(self.sub_dataset_lengths) if length == 0]
        if empty_dataset_indices:
            raise ValueError(f'child datasets at indices {empty_dataset_indices} have zero length')

        probabilities = [w / weight_sum for w in self.sampling_weights]

        original_total_size = int(self.epoch_size) if self.epoch_size is not None else sum(self.sub_dataset_lengths)
        if self.batch_size is not None:
            original_total_size = int(math.ceil(original_total_size / self.batch_size)) * self.batch_size

        if self.ratio_mode == 'epoch':
            self.num_samples_per_sub_dataset = self._allocate_counts(probabilities, original_total_size)
            self.num_samples_per_batch = None
        else:
            num_batches = original_total_size // self.batch_size
            self.num_samples_per_batch = self._allocate_counts(probabilities, self.batch_size)
            self.num_samples_per_sub_dataset = [count * num_batches for count in self.num_samples_per_batch]

        self.total_size = sum(self.num_samples_per_sub_dataset)

        cumulative_sizes = [0] * len(self.dataset.datasets)
        for i in range(1, len(self.dataset.datasets)):
            cumulative_sizes[i] = cumulative_sizes[i - 1] + len(self.dataset.datasets[i - 1])
        self.offsets = cumulative_sizes

    def _allocate_counts(self, probabilities: list[float], total_size: int) -> list[int]:
        # Allocate integer sample counts while preserving the desired total size.
        float_counts = [p * total_size for p in probabilities]
        counts = [int(math.floor(v)) for v in float_counts]
        remain = total_size - sum(counts)

        if remain > 0:
            frac_order = sorted(range(len(float_counts)), key=lambda i: float_counts[i] - counts[i], reverse=True)
            for i in frac_order[:remain]:
                counts[i] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_size

    def _sample_local_indices(self, data_size: int, num_samples: int) -> np.ndarray:
        if num_samples == 0:
            return np.array([], dtype=np.int64)

        if self.shuffle:
            replace = num_samples > data_size
            return np.random.choice(data_size, num_samples, replace=replace)

        if num_samples > data_size:
            base_indices = np.arange(data_size)
            num_repeats = int(np.ceil(num_samples / data_size))
            return np.tile(base_indices, num_repeats)[:num_samples]

        return np.arange(num_samples)

    def _build_epoch_indices(self) -> np.ndarray:
        all_indices = []
        for i in range(len(self.dataset.datasets)):
            local_indices = self._sample_local_indices(self.sub_dataset_lengths[i], self.num_samples_per_sub_dataset[i])
            if len(local_indices) == 0:
                continue
            all_indices.append(local_indices + self.offsets[i])

        if not all_indices:
            indices = np.array([], dtype=np.int64)
        else:
            indices = np.concatenate(all_indices)

        if self.shuffle and len(indices) > 0:
            np.random.shuffle(indices)
        return indices

    def _build_batch_indices(self) -> np.ndarray:
        num_batches = self.total_size // self.batch_size
        local_index_pools = [
            self._sample_local_indices(self.sub_dataset_lengths[i], self.num_samples_per_sub_dataset[i])
            for i in range(len(self.dataset.datasets))
        ]
        pool_offsets = [0] * len(self.dataset.datasets)
        batches = []

        for _ in range(num_batches):
            batch_parts = []
            for i in range(len(self.dataset.datasets)):
                num_samples = self.num_samples_per_batch[i]
                if num_samples == 0:
                    continue
                start = pool_offsets[i]
                end = start + num_samples
                batch_parts.append(local_index_pools[i][start:end] + self.offsets[i])
                pool_offsets[i] = end

            batch_indices = np.concatenate(batch_parts)
            if self.shuffle and len(batch_indices) > 0:
                batch_indices = np.random.permutation(batch_indices)
            batches.append(batch_indices)

        if self.shuffle and len(batches) > 0:
            batch_order = np.random.permutation(len(batches))
            batches = [batches[i] for i in batch_order]

        if not batches:
            return np.array([], dtype=np.int64)
        return np.concatenate(batches)

    def _build_chunk_descriptors_for_dataset(self, dataset_index: int, num_samples: int) -> list[tuple[int, int, int]]:
        if num_samples == 0:
            return []

        data_size = self.sub_dataset_lengths[dataset_index]
        descriptors = []
        produced = 0
        num_chunks = int(math.ceil(data_size / self.chunk_size))

        while produced < num_samples:
            chunk_ids = np.arange(num_chunks, dtype=np.int64)
            if self.shuffle:
                np.random.shuffle(chunk_ids)

            for chunk_id in chunk_ids:
                start = int(chunk_id) * self.chunk_size
                end = min(start + self.chunk_size, data_size)
                count = min(end - start, num_samples - produced)
                if count <= 0:
                    continue
                descriptors.append((dataset_index, start, count))
                produced += count
                if produced >= num_samples:
                    break

        return descriptors

    def _iter_chunk_indices(self) -> Iterator[int]:
        descriptors = []
        for i, num_samples in enumerate(self.num_samples_per_sub_dataset):
            descriptors.extend(self._build_chunk_descriptors_for_dataset(i, num_samples))

        if self.shuffle and descriptors:
            order = np.random.permutation(len(descriptors))
            descriptors = [descriptors[i] for i in order]

        for dataset_index, start, count in descriptors:
            local_indices = np.arange(start, start + count, dtype=np.int64)
            if self.shuffle and self.shuffle_within_chunk and len(local_indices) > 1:
                local_indices = np.random.permutation(local_indices)
            yield from (local_indices + self.offsets[dataset_index]).tolist()

    def __iter__(self) -> Iterator[int]:
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1

            if self.index_mode == 'chunk':
                yield from self._iter_chunk_indices()
            elif self.ratio_mode == 'epoch':
                indices = self._build_epoch_indices()
                yield from indices.tolist()
            else:
                indices = self._build_batch_indices()
                yield from indices.tolist()

            if not self.infinite:
                break
