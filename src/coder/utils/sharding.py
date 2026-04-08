from __future__ import annotations

from typing import Iterable, List, Sequence, TypeVar


T = TypeVar("T")


def validate_shard_args(num_shards: int, shard_idx: int) -> None:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError("--shard_idx must satisfy 0 <= shard_idx < num_shards")


def take_shard(items: Sequence[T] | Iterable[T], num_shards: int, shard_idx: int) -> List[T]:
    validate_shard_args(num_shards=num_shards, shard_idx=shard_idx)
    if num_shards == 1:
        return list(items)
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_idx]
