# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import os
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import torch


ProfilePhase = Literal["prefill", "decode", "warmup"]
EPDirection = Literal["dispatch", "combine"]


@dataclass
class _MoELayerCall:
    call_id: int
    ep_size: int
    num_experts: int
    placement: str
    activation_width: int
    activation_element_size: int
    has_shared_experts: bool
    ep_group_ranks: list[int]
    routes: torch.Tensor | None = None
    route_source_sizes: list[int] | None = None
    route_logical_source_sizes: list[int] | None = None
    route_group_ranks: list[int] | None = None


def _expert_owner_table(
    num_experts: int, ep_size: int, placement: str
) -> torch.Tensor:
    if placement == "round_robin":
        return torch.arange(num_experts, dtype=torch.int64) % ep_size
    if placement != "linear":
        raise ValueError(f"Unsupported expert placement: {placement}")

    base, remainder = divmod(num_experts, ep_size)
    owners = [
        rank
        for rank in range(ep_size)
        for _ in range(base + (1 if rank < remainder else 0))
    ]
    return torch.tensor(owners, dtype=torch.int64)


def _pack_masks(masks: list[int], rank_count: int) -> bytes:
    mask_width = (rank_count + 7) // 8
    full_mask = (1 << rank_count) - 1
    if not masks or any(mask <= 0 or mask > full_mask for mask in masks):
        raise ValueError("communication masks must be non-zero group-local masks")
    return b"".join(mask.to_bytes(mask_width, "little") for mask in masks)


def _packed_pattern(
    source_masks: list[int], destination_masks: list[int], rank_count: int
) -> tuple[bytes, bytes | None]:
    source_blob = _pack_masks(source_masks, rank_count)
    full_mask = (1 << rank_count) - 1
    destination_blob = (
        None
        if all(mask == full_mask for mask in destination_masks)
        else _pack_masks(destination_masks, rank_count)
    )
    return source_blob, destination_blob


def _expert_bits(
    *,
    num_experts: int,
    ep_size: int,
    ep_group_ranks: list[int],
    collective_group_ranks: list[int],
    placement: str,
) -> list[int]:
    if len(ep_group_ranks) != ep_size:
        raise ValueError("EP rank metadata does not match EP size")
    group_positions = {
        global_rank: index
        for index, global_rank in enumerate(collective_group_ranks)
    }
    owners = _expert_owner_table(num_experts, ep_size, placement).tolist()
    return [
        1 << group_positions[ep_group_ranks[owner]]
        if ep_group_ranks[owner] in group_positions
        else 0
        for owner in owners
    ]


def _route_source_mask(route: list[int], expert_bits: list[int]) -> int:
    mask = 0
    for expert in route:
        if 0 <= expert < len(expert_bits):
            mask |= expert_bits[expert]
    return mask


def _ep_pattern(
    routes: torch.Tensor,
    source_sizes: list[int],
    logical_source_sizes: list[int],
    direction: EPDirection,
    num_experts: int,
    ep_size: int,
    ep_group_ranks: list[int],
    collective_group_ranks: list[int],
    placement: str,
    bytes_per_row: int,
) -> tuple[int, bytes, bytes | None]:
    total_rows = sum(source_sizes)
    if (
        routes.ndim != 2
        or routes.shape[0] < total_rows
        or len(source_sizes) != len(logical_source_sizes)
        or len(source_sizes) != len(collective_group_ranks)
        or len(ep_group_ranks) != ep_size
    ):
        raise ValueError("MoE routes and EP size metadata do not match")

    expert_bits = _expert_bits(
        num_experts=num_experts,
        ep_size=ep_size,
        ep_group_ranks=ep_group_ranks,
        collective_group_ranks=collective_group_ranks,
        placement=placement,
    )
    route_rows = routes[:total_rows].to(dtype=torch.int64).tolist()
    source_masks: list[int] = []
    destination_masks: list[int] = []
    start = 0
    for source_rank, (size, logical_size) in enumerate(
        zip(source_sizes, logical_source_sizes)
    ):
        for route in route_rows[start : start + logical_size]:
            owner_mask = _route_source_mask(route, expert_bits)
            if owner_mask == 0:
                continue
            source_bit = 1 << source_rank
            if direction == "dispatch":
                source_masks.append(source_bit)
                destination_masks.append(owner_mask)
            else:
                source_masks.append(owner_mask)
                destination_masks.append(source_bit)
        start += size

    source_blob, destination_blob = _packed_pattern(
        source_masks, destination_masks, len(collective_group_ranks)
    )
    return bytes_per_row, source_blob, destination_blob


class CCLProfiler:
    """Low-overhead CUDA-event profiler for the evaluation flow."""

    def __init__(self) -> None:
        self._event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._metadata_slots: list[dict[str, Any]] = []
        self._ended_slots: list[bool] = []
        self._layers: dict[int, _MoELayerCall] = {}
        self._iteration: dict[str, Any] | None = None
        self._rank = -1
        self._used_events = 0
        self._forward_id = -1
        self._profiled_forwards = 0
        self._prefill_seen = False
        self._forward_active = False
        self._current_layer_call_id: int | None = None
        self._current_ep_direction: EPDirection | None = None
        self._current_ep_logical_source_sizes: list[int] | None = None
        self._current_moe_all_reduce_width: int | None = None
        self._count_only = False

    @property
    def iteration_active(self) -> bool:
        return self._iteration is not None

    @property
    def forward_active(self) -> bool:
        return self._forward_active

    def start_iteration(
        self,
        *,
        phase: ProfilePhase,
        rank: int,
        batch_size: int | None = None,
        input_len: int | None = None,
        output_len: int | None = None,
        max_events: int | None = None,
    ) -> None:
        if self.iteration_active:
            raise RuntimeError("A CCL profiling iteration is already active")
        if phase not in ("prefill", "decode", "warmup"):
            raise ValueError(f"Unknown CCL profiling phase: {phase}")
        if phase != "warmup" and (
            batch_size is None or input_len is None or output_len is None
        ):
            raise ValueError(
                "Measured iterations require batch_size/input_len/output_len"
            )
        if phase == "prefill" and output_len != 1:
            raise ValueError("Prefill profiling requires output_len=1")

        # An explicit zero runs an untimed structural calibration pass.
        count_only = max_events == 0
        if not count_only:
            if max_events is None:
                env_max_events = os.getenv("VLLM_CCL_PROFILE_MAX_EVENTS")
                if env_max_events is not None:
                    max_events = int(env_max_events)
                elif self._event_pairs:
                    max_events = len(self._event_pairs)
                else:
                    raise ValueError(
                        "max_events or VLLM_CCL_PROFILE_MAX_EVENTS is required"
                    )
            if max_events <= 0:
                raise ValueError("max_events must be positive")
            self._ensure_event_pool(max_events)

        self._iteration = {
            "phase": phase,
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
        }
        self._rank = rank
        self._used_events = 0
        self._forward_id = -1
        self._profiled_forwards = 0
        self._prefill_seen = False
        self._forward_active = False
        self._current_layer_call_id = None
        self._current_ep_direction = None
        self._current_ep_logical_source_sizes = None
        self._current_moe_all_reduce_width = None
        self._count_only = count_only
        self._layers.clear()

    def _ensure_event_pool(self, max_events: int) -> None:
        missing = max_events - len(self._event_pairs)
        self._event_pairs.extend(
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(max(0, missing))
        )
        self._metadata_slots.extend({} for _ in range(max(0, missing)))
        self._ended_slots.extend(False for _ in range(max(0, missing)))

    def select_forward(self, num_scheduled_tokens: dict[str, int]) -> bool:
        if not self.iteration_active:
            return False
        assert self._iteration is not None
        phase: ProfilePhase = self._iteration["phase"]
        if phase == "warmup":
            return True

        batch_size = self._iteration["batch_size"]
        input_len = self._iteration["input_len"]
        assert batch_size is not None and input_len is not None
        lengths = [int(value) for value in num_scheduled_tokens.values()]
        if len(lengths) != batch_size:
            raise AssertionError(
                f"Profiled {phase} forward has {len(lengths)} requests, "
                f"expected {batch_size}"
            )

        if phase == "prefill":
            if self._prefill_seen:
                raise AssertionError("Prefill profiling observed more than one forward")
            if any(length != input_len for length in lengths):
                raise AssertionError(
                    f"Profiled prefill must schedule {input_len} tokens per request: "
                    f"{lengths}"
                )
            self._prefill_seen = True
            return True

        if not self._prefill_seen:
            if any(length != input_len for length in lengths):
                raise AssertionError(
                    "Decode profiling expected one complete, unprofiled prefill "
                    f"of {input_len} tokens per request: {lengths}"
                )
            self._prefill_seen = True
            return False
        if any(length != 1 for length in lengths):
            raise AssertionError(
                f"Profiled decode must schedule one token per request: {lengths}"
            )
        return True

    def begin_forward(self) -> int:
        if self._forward_active:
            raise RuntimeError("Nested profiled forwards are not supported")
        self._forward_id += 1
        self._profiled_forwards += 1
        self._forward_active = True
        return self._begin_event(
            {
                "op": "forward",
                "forward_id": self._forward_id,
            }
        )

    def end_forward(self, token: int) -> None:
        self._end_event(token)
        self._forward_active = False
        self._current_layer_call_id = None
        self._current_ep_direction = None
        self._current_ep_logical_source_sizes = None
        self._current_moe_all_reduce_width = None

    def _begin_event(self, metadata: dict[str, Any]) -> int:
        if self._count_only:
            slot = self._used_events
            self._used_events += 1
            return slot
        if self._used_events >= len(self._event_pairs):
            raise RuntimeError(
                "CCL profiler event pool exhausted; increase "
                "VLLM_CCL_PROFILE_MAX_EVENTS or max_events"
            )
        slot = self._used_events
        self._used_events += 1
        metadata_slot = self._metadata_slots[slot]
        metadata_slot.clear()
        metadata_slot.update(metadata)
        self._ended_slots[slot] = False
        self._event_pairs[slot][0].record()
        return slot

    def _end_event(self, token: int) -> None:
        if self._count_only:
            return
        self._event_pairs[token][1].record()
        self._ended_slots[token] = True

    def begin_collective(
        self, metadata: dict[str, Any] | None
    ) -> int | None:
        if not self._forward_active or metadata is None:
            return None
        if self._count_only:
            return self._begin_event({})
        metadata["forward_id"] = self._forward_id
        return self._begin_event(metadata)

    def end_collective(self, token: int | None) -> None:
        if token is not None:
            self._end_event(token)

    @contextmanager
    def moe_layer(
        self,
        *,
        ep_size: int,
        num_experts: int,
        placement: str,
        activation_width: int,
        activation_element_size: int,
        has_shared_experts: bool,
        ep_group_ranks: list[int],
    ):
        if self._count_only:
            previous = self._current_layer_call_id
            self._current_layer_call_id = -1
            try:
                yield
            finally:
                self._current_layer_call_id = previous
            return
        call_id = len(self._layers)
        self._layers[call_id] = _MoELayerCall(
            call_id=call_id,
            ep_size=int(ep_size),
            num_experts=int(num_experts),
            placement=placement,
            activation_width=int(activation_width),
            activation_element_size=int(activation_element_size),
            has_shared_experts=has_shared_experts,
            ep_group_ranks=[int(rank) for rank in ep_group_ranks],
        )
        previous = self._current_layer_call_id
        self._current_layer_call_id = call_id
        try:
            yield
        finally:
            self._current_layer_call_id = previous

    def capture_moe_routes(self, topk_ids: torch.Tensor) -> None:
        if (
            self._count_only
            or not self._forward_active
            or self._current_layer_call_id is None
        ):
            return
        self._layers[self._current_layer_call_id].routes = topk_ids.detach()

    @contextmanager
    def ep_collective(
        self,
        direction: EPDirection,
        logical_source_sizes: list[int] | None = None,
    ):
        previous_direction = self._current_ep_direction
        previous_sizes = self._current_ep_logical_source_sizes
        self._current_ep_direction = direction
        self._current_ep_logical_source_sizes = (
            list(logical_source_sizes)
            if logical_source_sizes is not None
            else None
        )
        try:
            yield
        finally:
            self._current_ep_direction = previous_direction
            self._current_ep_logical_source_sizes = previous_sizes

    @contextmanager
    def moe_final_all_reduce(self, logical_width: int):
        previous = self._current_moe_all_reduce_width
        self._current_moe_all_reduce_width = int(logical_width)
        try:
            yield
        finally:
            self._current_moe_all_reduce_width = previous

    def all_reduce_metadata(
        self,
        tensor: torch.Tensor,
        *,
        group_ranks: list[int],
        rank_in_group: int,
    ) -> dict[str, Any] | None:
        if not self._forward_active:
            return None
        if self._count_only:
            return {}
        metadata = {
            "op": "all_reduce",
            "group_ranks": list(group_ranks),
            "rank_in_group": rank_in_group,
        }
        if self._current_moe_all_reduce_width is not None:
            assert self._current_layer_call_id is not None
            metadata["layer_call_id"] = self._current_layer_call_id
            metadata["logical_width"] = self._current_moe_all_reduce_width
            metadata["tensor_rows"] = prod(tensor.shape[:-1])
            metadata["element_size"] = tensor.element_size()
        else:
            metadata["input_bytes"] = int(tensor.numel() * tensor.element_size())
        return metadata

    def ep_metadata(
        self,
        *,
        sizes: list[int],
        group_ranks: list[int],
        rank_in_group: int,
    ) -> dict[str, Any] | None:
        if (
            not self._forward_active
            or self._current_ep_direction is None
            or self._current_layer_call_id is None
        ):
            return None
        if self._count_only:
            return {}
        layer = self._layers[self._current_layer_call_id]
        if len(sizes) != len(group_ranks):
            raise AssertionError("EP size metadata does not match communicator ranks")
        sizes = [int(size) for size in sizes]
        logical_source_sizes = (
            self._current_ep_logical_source_sizes
            if self._current_ep_logical_source_sizes is not None
            else sizes
        )
        if len(logical_source_sizes) != len(group_ranks):
            raise AssertionError("Logical EP sizes do not match communicator ranks")
        if self._current_ep_direction == "dispatch":
            layer.route_source_sizes = list(sizes)
            layer.route_logical_source_sizes = list(logical_source_sizes)
            layer.route_group_ranks = list(group_ranks)

        return {
            "op": "all_to_all",
            "direction": self._current_ep_direction,
            "group_ranks": list(group_ranks),
            "rank_in_group": rank_in_group,
            "layer_call_id": layer.call_id,
            "source_sizes": sizes,
            "logical_source_sizes": list(logical_source_sizes),
        }

    def finish_iteration(self) -> dict[str, Any]:
        if not self.iteration_active:
            raise RuntimeError("No CCL profiling iteration is active")
        if self._forward_active:
            raise RuntimeError("Cannot finish CCL profiling during a model forward")
        assert self._iteration is not None
        self._validate_forward_count()

        count_only = self._count_only
        if self._used_events and not count_only:
            if not all(self._ended_slots[: self._used_events]):
                raise RuntimeError("CCL profiling has unfinished CUDA event slots")
            torch.cuda.synchronize()
        records = []
        if not count_only:
            route_cache: dict[int, torch.Tensor | None] = {}
            records = [
                self._resolve_slot(slot, route_cache)
                for slot in range(self._used_events)
            ]
        result = {
            "rank": self._rank,
            "records": records,
            "events_used": self._used_events,
        }

        self._iteration = None
        self._layers.clear()
        self._forward_active = False
        self._current_ep_logical_source_sizes = None
        self._count_only = False
        return result

    def _validate_forward_count(self) -> None:
        assert self._iteration is not None
        phase = self._iteration["phase"]
        if phase == "warmup":
            return
        if not self._prefill_seen:
            raise AssertionError(f"Profiled {phase} iteration did not execute prefill")
        if phase == "prefill":
            expected = 1
        else:
            output_len = self._iteration["output_len"]
            assert output_len is not None
            expected = max(output_len - 1, 0)
        if self._profiled_forwards != expected:
            raise AssertionError(
                f"Profiled {phase} iteration observed {self._profiled_forwards} "
                f"forwards, expected {expected}"
            )

    def _resolve_slot(
        self,
        slot: int,
        route_cache: dict[int, torch.Tensor | None],
    ) -> dict[str, Any]:
        start, end = self._event_pairs[slot]
        record = dict(self._metadata_slots[slot])
        elapsed_ms = start.elapsed_time(end)
        if record["op"] == "forward":
            return {
                "op": "forward",
                "forward_id": record["forward_id"],
                "elapsed_ms": elapsed_ms,
            }
        result = {
            "op": record["op"],
            "forward_id": record["forward_id"],
            "group_ranks": record["group_ranks"],
            "elapsed_ms": elapsed_ms,
        }
        if record["rank_in_group"] != 0:
            return result

        if record["op"] == "all_reduce":
            pattern = self._resolve_all_reduce(record, route_cache)
        else:
            assert record["op"] == "all_to_all"
            pattern = self._resolve_ep(record, route_cache)
        bytes_per_row, source_masks, destination_masks = pattern
        result["bytes_per_row"] = bytes_per_row
        result["source_masks"] = base64.b64encode(source_masks).decode("ascii")
        if destination_masks is not None:
            result["destination_masks"] = base64.b64encode(
                destination_masks
            ).decode("ascii")
        return result

    def _cpu_routes(
        self, call_id: int, cache: dict[int, torch.Tensor | None]
    ) -> torch.Tensor | None:
        if call_id not in cache:
            routes = self._layers[call_id].routes
            cache[call_id] = routes.to(device="cpu") if routes is not None else None
        return cache[call_id]

    def _resolve_all_reduce(
        self, record: dict[str, Any], route_cache: dict[int, torch.Tensor | None]
    ) -> tuple[int, bytes, bytes | None]:
        ranks = record["group_ranks"]
        full_mask = (1 << len(ranks)) - 1
        if "logical_width" not in record:
            source_masks, destination_masks = _packed_pattern(
                [full_mask], [full_mask], len(ranks)
            )
            return record["input_bytes"], source_masks, destination_masks

        call_id = record["layer_call_id"]
        layer = self._layers[call_id]
        routes = self._cpu_routes(call_id, route_cache)
        if routes is None or routes.ndim != 2:
            raise RuntimeError("MoE routes were not captured for final AllReduce")

        tensor_rows = record["tensor_rows"]
        if (
            layer.route_source_sizes is not None
            and layer.route_logical_source_sizes is not None
            and layer.route_group_ranks is not None
        ):
            if self._rank not in layer.route_group_ranks:
                raise RuntimeError("MoE route ranks do not include this rank")
            source_rank = layer.route_group_ranks.index(self._rank)
            route_start = sum(layer.route_source_sizes[:source_rank])
            route_count = layer.route_logical_source_sizes[source_rank]
            if (
                route_count > tensor_rows
                or routes.shape[0] < route_start + route_count
            ):
                raise RuntimeError("MoE routes do not match final AllReduce rows")
            routes = routes[route_start : route_start + route_count]
        else:
            if routes.shape[0] < tensor_rows:
                raise RuntimeError("MoE routes do not match final AllReduce rows")
            routes = routes[:tensor_rows]

        route_rows = routes.to(dtype=torch.int64).tolist()
        if layer.has_shared_experts or layer.ep_size == 1:
            source_mask_values = [
                full_mask
                for route in route_rows
                if any(0 <= expert < layer.num_experts for expert in route)
            ]
        else:
            expert_bits = _expert_bits(
                num_experts=layer.num_experts,
                ep_size=layer.ep_size,
                ep_group_ranks=layer.ep_group_ranks,
                collective_group_ranks=ranks,
                placement=layer.placement,
            )
            source_mask_values = [
                mask
                for route in route_rows
                if (mask := _route_source_mask(route, expert_bits)) != 0
            ]
        source_masks, destination_masks = _packed_pattern(
            source_mask_values,
            [full_mask] * len(source_mask_values),
            len(ranks),
        )
        return (
            record["logical_width"] * record["element_size"],
            source_masks,
            destination_masks,
        )

    def _resolve_ep(
        self, record: dict[str, Any], route_cache: dict[int, torch.Tensor | None]
    ) -> tuple[int, bytes, bytes | None]:
        call_id = record["layer_call_id"]
        layer = self._layers[call_id]
        routes = self._cpu_routes(call_id, route_cache)
        if routes is None:
            raise RuntimeError("MoE routes were not captured for AlltoAll")
        return _ep_pattern(
            routes,
            record["source_sizes"],
            record["logical_source_sizes"],
            record["direction"],
            layer.num_experts,
            layer.ep_size,
            layer.ep_group_ranks,
            record["group_ranks"],
            layer.placement,
            layer.activation_width * layer.activation_element_size,
        )


_CCL_PROFILER = CCLProfiler()


def get_ccl_profiler() -> CCLProfiler:
    return _CCL_PROFILER
