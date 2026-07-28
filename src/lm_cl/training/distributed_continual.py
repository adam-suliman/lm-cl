from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lm_cl.config import ContinualExperimentConfig, ContinualTaskConfig
from lm_cl.data import TokenBatch, TokenPosition
from lm_cl.metrics import JsonlMetricLogger
from lm_cl.training.checkpoint import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    sha256_file,
)
from lm_cl.training.continual import (
    ContinualTrainer,
    TrainerState,
    TrainingResult,
    _position_dict,
    tensor_norm,
)
from lm_cl.training.distributed import (
    DISTRIBUTED_SCHEMA_VERSION,
    DistributedContext,
    all_gather_int,
    all_reduce_float,
    all_reduce_int,
    assert_digest_equal,
    collective_raise_if_any,
    iter_partitioned_batches,
)


class DistributedContinualTrainer(ContinualTrainer):
    """Single-node DDP realization of the approved global logical batch."""

    def __init__(
        self,
        config: ContinualExperimentConfig,
        context: DistributedContext,
    ):
        if config.distributed is None:
            raise ValueError("Distributed trainer requires distributed config")
        self.distributed = context
        self.rank_logger: JsonlMetricLogger | None = None
        super().__init__(config)
        if self.device.type != context.device.type:
            raise RuntimeError("Trainer and distributed devices disagree")
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": (
                config.distributed.ddp_broadcast_buffers
            ),
            "find_unused_parameters": (
                config.distributed.ddp_find_unused_parameters
            ),
            "gradient_as_bucket_view": False,
        }
        if context.device.type == "cuda":
            ddp_kwargs.update(
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        self.ddp_model = DistributedDataParallel(self.model, **ddp_kwargs)
        self._checkpoint_rank_rng_states: list[dict[str, Any]] | None = None
        self._checkpoint_rank_topology: list[dict[str, Any]] | None = None
        self._checkpoint_state_digests: dict[str, str] | None = None

    def _prepare_output(self, *, resume: bool) -> None:
        outcome: list[dict[str, str] | None] = [None]
        if self.distributed.is_primary:
            try:
                super()._prepare_output(resume=resume)
                outcome[0] = {"status": "ok"}
            except BaseException as exc:
                outcome[0] = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(outcome, src=0)
        assert outcome[0] is not None
        if outcome[0]["status"] != "ok":
            raise RuntimeError(
                "Rank-0 output preparation failed: "
                + outcome[0]["message"]
            )
        dist.barrier()
        if (
            self.config.distributed is not None
            and self.config.distributed.per_rank_diagnostic_logs
        ):
            diagnostics_dir = self.output_dir / "rank-diagnostics"
            if self.distributed.is_primary:
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            self.rank_logger = JsonlMetricLogger(
                diagnostics_dir
                / f"rank-{self.distributed.rank:04d}.jsonl"
            )

    def _log(self, event: str, **values: Any) -> None:
        if self.distributed.is_primary:
            super()._log(
                event,
                distributed=True,
                distributed_backend=self.distributed.backend,
                world_size=self.distributed.world_size,
                global_rank=self.distributed.rank,
                local_rank=self.distributed.local_rank,
                global_logical_batch_size=(
                    self.config.optimization
                    .global_sequences_per_logical_batch
                ),
                reduction_policy=(
                    self.config.distributed.reduction_policy
                    if self.config.distributed is not None
                    else None
                ),
                **values,
            )
        if self.rank_logger is not None:
            self.rank_logger.log(
                {
                    "event": event,
                    "rank": self.distributed.rank,
                    "local_rank": self.distributed.local_rank,
                    "world_size": self.distributed.world_size,
                    "backend": self.distributed.backend,
                    "global_logical_batches": (
                        self.state.global_logical_batches
                    ),
                    **values,
                }
            )

    def _batch_iterator(
        self,
        source: Any,
        task: ContinualTaskConfig,
        *,
        start: TokenPosition,
    ) -> Iterator[TokenBatch]:
        sequence_end = (
            None
            if task.train_sequence_prefix_count is None
            else task.train_sequence_offset_count
            + task.train_sequence_prefix_count
        )
        return iter_partitioned_batches(
            source,
            sequence_length=task.train_source.sequence_length,
            global_sequences_per_batch=(
                self.config.optimization.global_sequences_per_logical_batch
            ),
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
            start=start,
            sequence_prefix_count=sequence_end,
        )

    def _restore_gradients(
        self,
        gradients: dict[str, torch.Tensor | None],
    ) -> None:
        super()._restore_gradients(gradients)
        if (
            self.config.distributed is not None
            and self.config.distributed.debug_assert_synced
        ):
            actual = {
                "gradients": assert_digest_equal(
                self._gradient_state(),
                self.distributed,
                label="restored partial slow gradients",
                )
            }
            assert self.optimizer is not None and self.scheduler is not None
            actual["model"] = assert_digest_equal(
                self.model.state_dict(),
                self.distributed,
                label="restored model",
            )
            actual["optimizer"] = assert_digest_equal(
                self.optimizer.state_dict(),
                self.distributed,
                label="restored optimizer",
            )
            actual["scheduler"] = assert_digest_equal(
                self.scheduler.state_dict(),
                self.distributed,
                label="restored scheduler",
            )
            actual["scaler"] = assert_digest_equal(
                self.scaler.state_dict(),
                self.distributed,
                label="restored GradScaler",
            )
            actual["trainer"] = assert_digest_equal(
                self.state.to_dict(),
                self.distributed,
                label="restored trainer state",
            )
            actual["memory"] = assert_digest_equal(
                self._memory_checkpoint_state(),
                self.distributed,
                label="restored memory state",
            )
            expected = self._checkpoint_state_digests
            if expected is None:
                raise ValueError(
                    "Distributed checkpoint lacks expected state digests"
                )
            mismatches = sorted(
                key for key, value in actual.items()
                if expected.get(key) != value
            )
            if mismatches:
                raise ValueError(
                    "Distributed resume state digest mismatch: "
                    + ", ".join(mismatches)
                )

    def _forward_model(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int,
        evaluation_root: torch.Tensor | None = None,
    ):
        if not self.config.variant.memory_enabled:
            return self.ddp_model(
                input_ids,
                labels,
                ignore_index=ignore_index,
            )
        root = evaluation_root
        if root is None and self.config.variant.persistent_fast_memory:
            root = self._fast_effective_memory()
        return self.ddp_model(
            input_ids,
            labels,
            root_memory=root,
            ignore_index=ignore_index,
        )

    def _stash_synced_window_gradients(
        self,
    ) -> dict[str, torch.Tensor | None]:
        prior: dict[str, torch.Tensor | None] = {}
        for name, parameter in self.model.named_parameters():
            prior[name] = (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            parameter.grad = None
        return prior

    def _restore_prior_window_gradients(
        self,
        prior: dict[str, torch.Tensor | None],
    ) -> None:
        for name, parameter in self.model.named_parameters():
            previous = prior[name]
            if previous is None:
                continue
            if parameter.grad is None:
                parameter.grad = previous
            else:
                parameter.grad.add_(previous)

    def _apply_distributed_fast_update(
        self,
        *,
        global_valid_target_count: int,
    ) -> dict[str, float]:
        if self.active_memory is None or self.active_memory.grad is None:
            raise RuntimeError("FastMem active root did not receive a gradient")
        gradient = self.active_memory.grad.detach().clone()
        if self.scaler.is_enabled():
            gradient.mul_(
                self._backward_reference_targets / self.scaler.get_scale()
            )
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(self.distributed.world_size)
        gradient.div_(global_valid_target_count)
        finite = int(torch.isfinite(gradient).all())
        if all_reduce_int(
            1 - finite,
            self.distributed,
            op=dist.ReduceOp.MAX,
        ):
            raise FloatingPointError(
                "Distributed active-memory gradient is non-finite"
            )
        before = float(gradient.double().norm().cpu())
        threshold = self.config.variant.fast_memory_grad_clip_norm
        if threshold is None:
            raise RuntimeError("FastMem requires a gradient clip threshold")
        scale = min(1.0, threshold / max(before, 1e-30))
        clipped = gradient * scale
        after = float(clipped.double().norm().cpu())
        if self.distributed.is_primary:
            with torch.no_grad():
                next_memory = self.active_memory - (
                    self.config.variant.fast_lr * clipped
                )
        else:
            next_memory = torch.empty_like(self.active_memory)
        dist.broadcast(next_memory, src=0)
        self.active_memory = (
            next_memory.detach().clone().requires_grad_(True)
        )
        active_norm = float(self.active_memory.detach().double().norm().cpu())
        self.state.global_fast_updates += 1
        self.state.task_fast_updates += 1
        self.state.last_active_memory_grad_norm = before
        self.state.last_active_memory_clipped_grad_norm = after
        self.state.last_active_memory_norm = active_norm
        self.state.fast_gradient_norm_history.append(before)
        self.state.fast_clipped_gradient_norm_history.append(after)
        self.state.fast_memory_norm_history.append(active_norm)
        if (
            self.config.distributed is not None
            and self.config.distributed.debug_assert_synced
        ):
            assert_digest_equal(
                self.active_memory,
                self.distributed,
                label="active memory after fast update",
            )
        return {
            "active_memory_gradient_norm_before_clip": before,
            "active_memory_gradient_norm_after_clip": after,
            "active_memory_norm": active_norm,
        }

    def _train_logical_batch(
        self,
        batch: TokenBatch,
        *,
        ignore_index: int,
    ) -> dict[str, Any]:
        if (
            batch.global_sequence_count is None
            or batch.local_slice_start is None
            or batch.local_slice_end is None
            or batch.global_sequence_start is None
            or batch.global_sequence_end is None
        ):
            raise RuntimeError("Distributed batch lacks global partition metadata")
        started = time.monotonic()
        input_ids = torch.from_numpy(batch.input_ids)
        labels = torch.from_numpy(batch.labels)
        local_examples = len(input_ids)
        microbatch = self.config.optimization.physical_microbatch_sequences
        maximum_local_examples = all_reduce_int(
            local_examples,
            self.distributed,
            op=dist.ReduceOp.MAX,
        )
        microbatch_slots = max(
            1,
            math.ceil(maximum_local_examples / microbatch),
        )
        prior_gradients = self._stash_synced_window_gradients()
        local_loss_sum = 0.0
        local_target_count = 0
        segment_count = (
            batch.input_ids.shape[1] // self.config.variant.segment_length
            if self.config.variant.memory_enabled
            else 0
        )
        local_segment_norm_sums = [0.0 for _ in range(segment_count)]
        local_segment_examples = 0
        self.model.train()
        for slot in range(microbatch_slots):
            start = slot * microbatch
            end = min(start + microbatch, local_examples)
            has_real_examples = start < end
            if has_real_examples:
                micro_inputs = input_ids[start:end].to(self.device)
                micro_labels = labels[start:end].to(self.device)
            else:
                sequence_length = batch.input_ids.shape[1]
                micro_inputs = torch.zeros(
                    (1, sequence_length),
                    dtype=torch.long,
                    device=self.device,
                )
                micro_labels = micro_inputs
            synchronization = (
                nullcontext()
                if slot + 1 == microbatch_slots
                else self.ddp_model.no_sync()
            )
            output = None
            local_error = None
            with synchronization:
                try:
                    with self._autocast():
                        output = self._forward_model(
                            micro_inputs,
                            micro_labels,
                            ignore_index=ignore_index,
                        )
                    if output.loss_sum is None:
                        raise RuntimeError(
                            "Distributed forward did not return loss"
                        )
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                collective_raise_if_any(
                    local_error,
                    self.distributed,
                    prefix="Distributed logical-batch forward failed",
                )
                assert output is not None and output.loss_sum is not None
                local_nonfinite = int(
                    has_real_examples
                    and not bool(torch.isfinite(output.loss_sum))
                )
                if all_reduce_int(
                    local_nonfinite,
                    self.distributed,
                    op=dist.ReduceOp.MAX,
                ):
                    raise FloatingPointError(
                        "A rank produced non-finite logical-batch loss"
                    )
                multiplier = (
                    float(self.distributed.world_size)
                    if has_real_examples
                    else 0.0
                )
                backward_loss = output.loss_sum * multiplier
                if self.scaler.is_enabled():
                    self.scaler.scale(
                        backward_loss / self._backward_reference_targets
                    ).backward()
                else:
                    backward_loss.backward()
            if has_real_examples:
                local_loss_sum += float(output.loss_sum.detach().cpu())
                local_target_count += int(output.target_count.detach().cpu())
                if output.segment_write_memories is not None:
                    if len(output.segment_write_memories) != segment_count:
                        raise RuntimeError(
                            "Model segment diagnostics disagree with "
                            "configured sequence segmentation"
                        )
                    for index, memory in enumerate(
                        output.segment_write_memories
                    ):
                        local_segment_norm_sums[index] += float(
                            memory.detach()
                            .double()
                            .flatten(1)
                            .norm(dim=1)
                            .sum()
                            .cpu()
                        )
                    local_segment_examples += end - start
        self._restore_prior_window_gradients(prior_gradients)
        if local_target_count != batch.valid_target_count:
            raise RuntimeError(
                "Rank-local valid-target count differs from source contract"
            )
        global_target_count = all_reduce_int(
            local_target_count,
            self.distributed,
        )
        global_input_tokens = all_reduce_int(
            int(batch.input_ids.size),
            self.distributed,
        )
        global_loss_sum = all_reduce_float(
            local_loss_sum,
            self.distributed,
        )
        expected_input_tokens = (
            batch.global_sequence_count * batch.input_ids.shape[1]
        )
        if global_input_tokens != expected_input_tokens:
            raise RuntimeError("Distributed input-token coverage is incomplete")
        if global_target_count <= 0:
            raise RuntimeError("Global logical batch has no valid targets")
        if self.config.variant.persistent_fast_memory:
            fast_metrics: dict[str, float | None] = (
                self._apply_distributed_fast_update(
                    global_valid_target_count=global_target_count
                )
            )
        else:
            fast_metrics = {
                "active_memory_gradient_norm_before_clip": None,
                "active_memory_gradient_norm_after_clip": None,
                "active_memory_norm": None,
            }
        global_segment_examples = all_reduce_int(
            local_segment_examples,
            self.distributed,
        )
        segment_norms = []
        for index, local_value in enumerate(local_segment_norm_sums):
            global_value = all_reduce_float(
                local_value,
                self.distributed,
            )
            segment_norms.append(
                {
                    "segment_index": index,
                    "encoded_write_memory_norm": (
                        global_value / global_segment_examples
                    ),
                }
            )
        final_write_norm = (
            None
            if not segment_norms
            else segment_norms[-1]["encoded_write_memory_norm"]
        )
        self.state.last_encoded_write_memory_norm = final_write_norm
        elapsed = max(time.monotonic() - started, 1e-12)
        self.state.logical_batch_within_task += 1
        self.state.global_logical_batches += 1
        self.state.global_input_tokens += global_input_tokens
        self.state.global_valid_targets += global_target_count
        self.state.task_input_tokens += global_input_tokens
        self.state.task_valid_targets += global_target_count
        self.state.task_loss_sum += global_loss_sum
        self.state.window_logical_batches += 1
        self.state.window_valid_targets += global_target_count
        self.state.window_loss_sum += global_loss_sum
        self.state.source_position = _position_dict(batch.next_position)
        mean_loss = global_loss_sum / global_target_count
        self.state.loss_history.append(mean_loss)
        per_rank_examples = all_gather_int(
            local_examples,
            self.distributed,
        )
        per_rank_targets = all_gather_int(
            local_target_count,
            self.distributed,
        )
        record = {
            "loss_sum": global_loss_sum,
            "valid_target_count": global_target_count,
            "mean_loss": mean_loss,
            "input_token_count": global_input_tokens,
            "throughput_input_tokens_per_second": (
                global_input_tokens / elapsed
            ),
            "parameter_norm": tensor_norm(self.model.parameters()),
            "gradient_norm": None,
            "learning_rate": (
                None if self.scheduler is None else self.scheduler.current_lr
            ),
            "source_start_position": _position_dict(batch.start_position),
            "source_position": self.state.source_position,
            "source_global_sequence_range": [
                batch.global_sequence_start,
                batch.global_sequence_end,
            ],
            "rank_partition_rule": self.distributed.partition_rule,
            "per_rank_example_counts": per_rank_examples,
            "per_rank_target_counts": per_rank_targets,
            "encoded_write_memory_norm": final_write_norm,
            "segment_memory_diagnostics": segment_norms,
            **fast_metrics,
        }
        self._log("logical_batch", **record)
        return record

    def _optimizer_step(self, *, tail_flush: bool) -> dict[str, Any]:
        assert self.optimizer is not None and self.scheduler is not None
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        local_nonfinite = int(
            any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in self.model.parameters()
            )
        )
        if all_reduce_int(
            local_nonfinite,
            self.distributed,
            op=dist.ReduceOp.MAX,
        ):
            raise FloatingPointError(
                "A rank produced a non-finite slow gradient"
            )
        gradient_norm = self._normalized_gradient_norm()
        learning_rate = self.scheduler.current_lr
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.state.task_slow_steps += 1
        self.state.global_slow_steps += 1
        result = {
            "loss_sum": self.state.window_loss_sum,
            "valid_target_count": self.state.window_valid_targets,
            "mean_loss": (
                self.state.window_loss_sum / self.state.window_valid_targets
            ),
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "parameter_norm": tensor_norm(self.model.parameters()),
            "window_logical_batches": self.state.window_logical_batches,
            "tail_flush": tail_flush,
        }
        self.optimizer.zero_grad(set_to_none=True)
        self.state.window_logical_batches = 0
        self.state.window_valid_targets = 0
        self.state.window_loss_sum = 0.0
        if (
            self.config.distributed is not None
            and self.config.distributed.debug_assert_synced
        ):
            assert_digest_equal(
                self.model.state_dict(),
                self.distributed,
                label="model after optimizer step",
            )
            assert_digest_equal(
                self.optimizer.state_dict(),
                self.distributed,
                label="optimizer after step",
            )
            assert_digest_equal(
                self.scheduler.state_dict(),
                self.distributed,
                label="scheduler after step",
            )
            assert_digest_equal(
                self.scaler.state_dict(),
                self.distributed,
                label="GradScaler after step",
            )
        self._log("optimizer_step", **result)
        return result

    def _distributed_checkpoint_metadata(self) -> dict[str, Any]:
        assert self.config.distributed is not None
        assert self._checkpoint_rank_rng_states is not None
        assert self._checkpoint_rank_topology is not None
        assert self._checkpoint_state_digests is not None
        return {
            "schema_version": DISTRIBUTED_SCHEMA_VERSION,
            "enabled": True,
            "backend": self.distributed.backend,
            "world_size": self.distributed.world_size,
            "rank_topology": self._checkpoint_rank_topology,
            "global_logical_batch_size": (
                self.config.optimization.global_sequences_per_logical_batch
            ),
            "partition_rule": self.distributed.partition_rule,
            "rank_rng_states": self._checkpoint_rank_rng_states,
            "global_source_position": self.state.source_position,
            "global_input_tokens": self.state.global_input_tokens,
            "global_valid_targets": self.state.global_valid_targets,
            "reduction_policy": self.config.distributed.reduction_policy,
            "ddp": {
                "broadcast_buffers": (
                    self.config.distributed.ddp_broadcast_buffers
                ),
                "find_unused_parameters": (
                    self.config.distributed.ddp_find_unused_parameters
                ),
                "gradient_as_bucket_view": False,
                "world_size_scaled_local_loss": True,
                "ddp_gradient_semantics": "average",
            },
            "active_memory_sync_policy": (
                self.config.distributed.active_memory_sync_policy
            ),
            "state_digests": self._checkpoint_state_digests,
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._checkpoint_payload()
        payload["distributed_state"] = (
            self._distributed_checkpoint_metadata()
        )
        return payload

    def _verify_shared_state_for_checkpoint(self) -> dict[str, str]:
        values = {
            "model": self.model.state_dict(),
            "gradients": self._gradient_state(),
            "optimizer": (
                None if self.optimizer is None else self.optimizer.state_dict()
            ),
            "scheduler": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "scaler": self.scaler.state_dict(),
            "trainer": self.state.to_dict(),
            "memory": self._memory_checkpoint_state(),
        }
        return {
            label: assert_digest_equal(
                value,
                self.distributed,
                label=f"checkpoint {label}",
            )
            for label, value in values.items()
        }

    def _save_checkpoint(self, filename: str) -> tuple[str, str]:
        self._log(
            "checkpoint_barrier",
            checkpoint_filename=filename,
            barrier_phase="before_shared_state_verification",
        )
        dist.barrier()
        self._checkpoint_state_digests = (
            self._verify_shared_state_for_checkpoint()
        )
        rng_states: list[dict[str, Any] | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(
            rng_states,
            {
                "rank": self.distributed.rank,
                "state": capture_rng_state(),
            },
        )
        topology: list[dict[str, Any] | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(
            topology,
            self.distributed.topology_record(),
        )
        self._checkpoint_rank_rng_states = [
            value for value in rng_states if value is not None
        ]
        self._checkpoint_rank_topology = [
            value for value in topology if value is not None
        ]
        outcome: list[dict[str, str] | None] = [None]
        path = self.checkpoint_dir / filename
        if self.distributed.is_primary:
            try:
                checksum = atomic_save_checkpoint(
                    path,
                    self._checkpoint_payload(),
                )
                outcome[0] = {
                    "status": "ok",
                    "path": str(path),
                    "sha256": checksum,
                }
            except BaseException as exc:
                outcome[0] = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(outcome, src=0)
        assert outcome[0] is not None
        if outcome[0]["status"] != "ok":
            raise RuntimeError(
                "Rank-0 checkpoint write failed: "
                + outcome[0]["message"]
            )
        dist.barrier()
        self._log(
            "checkpoint",
            checkpoint_path=outcome[0]["path"],
            checkpoint_sha256=outcome[0]["sha256"],
            source_position=self.state.source_position,
            writer_rank=0,
            barrier_phase="after_atomic_write",
        )
        return outcome[0]["path"], outcome[0]["sha256"]

    def _load_resume(
        self,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        identity = sha256_file(checkpoint_path)
        identities: list[str | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(identities, identity)
        if len(set(identities)) != 1:
            raise ValueError(
                "Ranks disagree about checkpoint identity: "
                f"{identities}"
            )
        if payload["config_sha256"] != self.config_sha256:
            raise ValueError("Resume configuration does not match checkpoint")
        metadata = payload.get("distributed_state")
        if not isinstance(metadata, dict) or not metadata.get("enabled"):
            raise ValueError("Distributed resume requires distributed metadata")
        if metadata.get("schema_version") != DISTRIBUTED_SCHEMA_VERSION:
            raise ValueError("Distributed resume schema-version mismatch")
        if metadata.get("backend") != self.distributed.backend:
            raise ValueError("Distributed resume backend mismatch")
        if metadata.get("world_size") != self.distributed.world_size:
            raise ValueError("Distributed resume world-size mismatch")
        if metadata.get("global_logical_batch_size") != (
            self.config.optimization.global_sequences_per_logical_batch
        ):
            raise ValueError(
                "Distributed resume global logical-batch-size mismatch"
            )
        if metadata.get("partition_rule") != self.distributed.partition_rule:
            raise ValueError("Distributed resume partition-rule mismatch")
        if metadata.get("reduction_policy") != (
            self.config.distributed.reduction_policy
        ):
            raise ValueError("Distributed resume reduction-policy mismatch")
        if metadata.get("active_memory_sync_policy") != (
            self.config.distributed.active_memory_sync_policy
        ):
            raise ValueError(
                "Distributed resume active-memory synchronization mismatch"
            )
        state_digests = metadata.get("state_digests")
        if not isinstance(state_digests, dict):
            raise ValueError("Distributed checkpoint lacks state digests")
        self._checkpoint_state_digests = state_digests
        rank_rng_states = metadata.get("rank_rng_states")
        if not isinstance(rank_rng_states, list):
            raise ValueError("Distributed checkpoint lacks rank RNG states")
        by_rank = {
            item.get("rank"): item.get("state")
            for item in rank_rng_states
            if isinstance(item, dict)
        }
        if set(by_rank) != set(range(self.distributed.world_size)):
            raise ValueError("Distributed checkpoint rank RNG state is incomplete")
        if metadata.get("global_source_position") != (
            payload["trainer_state"]["source_position"]
        ):
            raise ValueError(
                "Distributed checkpoint global source position is corrupt"
            )
        for counter in ("global_input_tokens", "global_valid_targets"):
            if metadata.get(counter) != payload["trainer_state"][counter]:
                raise ValueError(
                    f"Distributed checkpoint {counter} is corrupt"
                )
        self.model.load_state_dict(payload["model_state"])
        self.state = TrainerState.from_dict(payload["trainer_state"])
        self._restore_memory_state(payload["memory_state"])
        restore_rng_state(by_rank[self.distributed.rank])
        assert_digest_equal(
            self.model.state_dict(),
            self.distributed,
            label="resumed model",
        )
        if self.active_memory is not None:
            assert_digest_equal(
                self.active_memory,
                self.distributed,
                label="resumed active memory",
            )
        return payload

    def run(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        stop_after_global_logical_batches: int | None = None,
        stop_after_task_boundaries: int | None = None,
    ) -> TrainingResult:
        try:
            return super().run(
                resume_checkpoint=resume_checkpoint,
                stop_after_global_logical_batches=(
                    stop_after_global_logical_batches
                ),
                stop_after_task_boundaries=stop_after_task_boundaries,
            )
        finally:
            if self.rank_logger is not None:
                self.rank_logger.close()
