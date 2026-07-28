from __future__ import annotations

import hashlib
import math
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist

from lm_cl.config import DistributedConfig
from lm_cl.data.packed import PackedShardSource
from lm_cl.data.sources import (
    ArrayTokenSource,
    SyntheticBatchSource,
)
from lm_cl.data.types import TokenBatch, TokenPosition


PARTITION_RULE_VERSION = "contiguous_floor_v1"
DISTRIBUTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LogicalBatchPartition:
    global_size: int
    rank: int
    world_size: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def plan_logical_batch_partition(
    global_size: int,
    rank: int,
    world_size: int,
) -> LogicalBatchPartition:
    if global_size < 0:
        raise ValueError("global_size must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    start = math.floor(global_size * rank / world_size)
    end = math.floor(global_size * (rank + 1) / world_size)
    return LogicalBatchPartition(
        global_size=global_size,
        rank=rank,
        world_size=world_size,
        start=start,
        end=end,
    )


def all_partitions(
    global_size: int,
    world_size: int,
) -> list[LogicalBatchPartition]:
    return [
        plan_logical_batch_partition(global_size, rank, world_size)
        for rank in range(world_size)
    ]


@dataclass
class DistributedContext:
    enabled: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    timeout_seconds: int
    partition_rule: str

    @classmethod
    def initialize(
        cls,
        config: DistributedConfig,
        *,
        runtime_device: str,
    ) -> "DistributedContext":
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Distributed execution requires torchrun environment fields: "
                + ", ".join(missing)
            )
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(world_size))
        )
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise RuntimeError("Invalid torchrun global rank topology")
        if (
            local_world_size <= 0
            or local_rank < 0
            or local_rank >= local_world_size
        ):
            raise RuntimeError("Invalid torchrun local rank topology")
        use_cuda = runtime_device == "cuda" or (
            runtime_device == "auto" and torch.cuda.is_available()
        )
        backend = config.backend
        if backend == "auto":
            backend = "nccl" if use_cuda else "gloo"
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL requested but CUDA is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("LOCAL_RANK exceeds visible CUDA devices")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            if use_cuda:
                raise RuntimeError("CUDA distributed execution requires NCCL")
            device = torch.device("cpu")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=config.timeout_seconds),
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            dist.destroy_process_group()
            raise RuntimeError("Initialized process-group topology disagrees")
        return cls(
            enabled=True,
            backend=backend,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            device=device,
            timeout_seconds=config.timeout_seconds,
            partition_rule=config.partition_rule,
        )

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def topology_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "hostname": socket.gethostname(),
            "device": str(self.device),
        }
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            record.update(
                {
                    "device_name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_bytes": properties.total_memory,
                }
            )
        return record

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _collective_device(context: DistributedContext) -> torch.device:
    return context.device if context.backend == "nccl" else torch.device("cpu")


def all_reduce_int(
    value: int,
    context: DistributedContext,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> int:
    tensor = torch.tensor(
        value,
        dtype=torch.int64,
        device=_collective_device(context),
    )
    dist.all_reduce(tensor, op=op)
    return int(tensor.cpu())


def all_reduce_float(
    value: float,
    context: DistributedContext,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> float:
    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device=_collective_device(context),
    )
    dist.all_reduce(tensor, op=op)
    return float(tensor.cpu())


def all_gather_int(value: int, context: DistributedContext) -> list[int]:
    tensor = torch.tensor(
        value,
        dtype=torch.int64,
        device=_collective_device(context),
    )
    gathered = [torch.zeros_like(tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, tensor)
    return [int(item.cpu()) for item in gathered]


def collective_raise_if_any(
    local_error: str | None,
    context: DistributedContext,
    *,
    prefix: str,
) -> None:
    failed = all_reduce_int(
        int(local_error is not None),
        context,
        op=dist.ReduceOp.MAX,
    )
    if not failed:
        return
    errors: list[str | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(errors, local_error)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise RuntimeError(f"{prefix}: " + " | ".join(failures))


def _hash_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def assert_digest_equal(
    value: Any,
    context: DistributedContext,
    *,
    label: str,
) -> str:
    local = state_digest(value)
    values: list[str | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(values, local)
    if len(set(values)) != 1:
        raise RuntimeError(f"Distributed {label} differs across ranks: {values}")
    return local


def _empty_batch_arrays(sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    empty = np.empty((0, sequence_length), dtype=np.int64)
    return empty, empty.copy()


def iter_partitioned_batches(
    source: Any,
    *,
    sequence_length: int,
    global_sequences_per_batch: int,
    rank: int,
    world_size: int,
    start: TokenPosition | None = None,
    sequence_prefix_count: int | None = None,
) -> Iterator[TokenBatch]:
    if sequence_length <= 1 or global_sequences_per_batch <= 0:
        raise ValueError("Invalid distributed batch dimensions")
    position = start or TokenPosition(0, 0)
    if sequence_prefix_count is not None and sequence_prefix_count <= 0:
        raise ValueError("sequence_prefix_count must be positive")
    while True:
        if isinstance(source, SyntheticBatchSource):
            if sequence_length != source.config.sequence_length:
                raise ValueError("Synthetic sequence length mismatch")
            global_start = (
                len(source.dataset)
                if position.shard_index == 1
                else position.token_offset
            )
            available = len(source.dataset) - global_start
            if sequence_prefix_count is not None:
                available = min(
                    available,
                    max(sequence_prefix_count - global_start, 0),
                )
            global_count = min(global_sequences_per_batch, max(available, 0))
            if global_count == 0:
                return
            partition = plan_logical_batch_partition(
                global_count, rank, world_size
            )
            indices = range(
                global_start + partition.start,
                global_start + partition.end,
            )
            arrays = [
                source.dataset[index]["input_ids"].numpy()
                for index in indices
            ]
            label_arrays = [
                source.dataset[index]["labels"].numpy()
                for index in range(
                    global_start + partition.start,
                    global_start + partition.end,
                )
            ]
            if arrays:
                inputs = np.stack(arrays).astype(np.int64, copy=False)
                labels = np.stack(label_arrays).astype(np.int64, copy=False)
            else:
                inputs, labels = _empty_batch_arrays(sequence_length)
            global_end = global_start + global_count
            next_position = (
                TokenPosition(1, 0)
                if global_end == len(source.dataset)
                else TokenPosition(0, global_end)
            )
            valid_targets = int(
                np.count_nonzero(
                    labels[:, 1:] != source.config.ignore_index
                )
            )
        elif isinstance(source, PackedShardSource):
            global_offset = source.global_offset(position)
            if global_offset % sequence_length:
                raise ValueError("Batch start is not sequence-aligned")
            if sequence_prefix_count is not None:
                available_complete = source.token_count // sequence_length
                if sequence_prefix_count > available_complete:
                    raise ValueError(
                        "sequence_prefix_count exceeds available complete sequences"
                    )
            available = (source.token_count - global_offset) // sequence_length
            if sequence_prefix_count is not None:
                available = min(
                    available,
                    max(
                        sequence_prefix_count
                        - global_offset // sequence_length,
                        0,
                    ),
                )
            global_count = min(global_sequences_per_batch, max(available, 0))
            if global_count == 0:
                return
            partition = plan_logical_batch_partition(
                global_count, rank, world_size
            )
            global_start = global_offset // sequence_length
            local_position = source.position_at(
                global_offset + partition.start * sequence_length
            )
            flat, _ = source.read_tokens(
                partition.size * sequence_length,
                start=local_position,
            )
            inputs = np.asarray(
                flat.reshape(partition.size, sequence_length),
                dtype=np.int64,
            )
            labels = inputs.copy()
            next_position = source.position_at(
                global_offset + global_count * sequence_length
            )
            global_end = global_start + global_count
            valid_targets = partition.size * (sequence_length - 1)
        elif isinstance(source, ArrayTokenSource):
            global_offset = source._offset(position)
            if global_offset % sequence_length:
                raise ValueError("Batch start is not sequence-aligned")
            available = (len(source.tokens) - global_offset) // sequence_length
            if sequence_prefix_count is not None:
                available = min(
                    available,
                    max(
                        sequence_prefix_count
                        - global_offset // sequence_length,
                        0,
                    ),
                )
            global_count = min(global_sequences_per_batch, max(available, 0))
            if global_count == 0:
                return
            partition = plan_logical_batch_partition(
                global_count, rank, world_size
            )
            global_start = global_offset // sequence_length
            local_start = global_offset + partition.start * sequence_length
            count = partition.size * sequence_length
            inputs = np.asarray(
                source.tokens[local_start : local_start + count].reshape(
                    partition.size, sequence_length
                ),
                dtype=np.int64,
            )
            labels = inputs.copy()
            next_position = source._position(
                global_offset + global_count * sequence_length
            )
            global_end = global_start + global_count
            valid_targets = partition.size * (sequence_length - 1)
        else:
            raise TypeError(
                f"Unsupported distributed source type: {type(source).__name__}"
            )
        yield TokenBatch(
            input_ids=inputs,
            labels=labels,
            valid_target_count=valid_targets,
            start_position=position,
            next_position=next_position,
            global_sequence_count=global_count,
            local_slice_start=partition.start,
            local_slice_end=partition.end,
            global_sequence_start=global_start,
            global_sequence_end=global_end,
        )
        position = next_position
