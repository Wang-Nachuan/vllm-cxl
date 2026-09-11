# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod
from time import monotonic_ns
from typing import Any, Literal

import torch


ProfilePhase = Literal["prefill", "decode"]
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
    route_specs: list[_RouteSpec] = field(default_factory=list)
    route_offset: int | None = None
    route_buffer: torch.Tensor | None = None
    route_source_sizes: list[int] | None = None
    route_group_ranks: list[int] | None = None


@dataclass(frozen=True)
class _RouteSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass
class _CollectiveLayout:
    metadata: dict[str, Any]
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass
class _GraphLayout:
    cudagraph_mode: Any
    batch_descriptor: Any
    collectives: list[_CollectiveLayout] = field(default_factory=list)
    layers: list[_MoELayerCall] = field(default_factory=list)
    route_buffer: torch.Tensor | None = None
    captured: bool = False
    host_route_slots: list[_HostRouteSlot] = field(default_factory=list)


@dataclass
class _HostRouteSlot:
    buffer: torch.Tensor | None
    routes: dict[int, torch.Tensor]


@dataclass
class _LayoutPass:
    kind: Literal["prepare", "capture"]
    layout: _GraphLayout
    collective_index: int = 0
    layer_index: int = 0
    current_layer_call_id: int | None = None
    current_ep_direction: EPDirection | None = None
    current_ep_logical_source_sizes: list[int] | None = None
    deferred_moe_all_reduce: tuple[int, int] | None = None
    open_collective: int | None = None
    route_indices: dict[int, int] = field(default_factory=dict)


@dataclass
class _PendingForward:
    token: int
    layout: _GraphLayout
    start: torch.cuda.Event
    end: torch.cuda.Event
    num_tokens_unpadded: int
    logical_source_sizes: list[int]
    host_route_buffer: torch.Tensor | None
    host_routes: dict[int, torch.Tensor]
    host_start_ns: int
    ended: bool = False
    routes_enqueued: bool = False


@dataclass
class _DrainedForward:
    forward_id: int
    layout: _GraphLayout
    elapsed_ms: float
    collective_times: list[tuple[float, float]]
    num_tokens_unpadded: int
    logical_source_sizes: list[int]
    host_routes: dict[int, torch.Tensor]
    host_start_ns: int


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
    """CUDA-graph-aware profiler for logical PyNccl collectives.

    A layout is prepared during an ordinary dummy pass, then populated with
    external CUDA event nodes during capture. Runtime graph replays update the
    layout's persistent events and route buffers. The caller must drain every
    profiled replay after vLLM's existing scheduler synchronization and before
    replaying the same graph again.
    """

    def __init__(self) -> None:
        self._layouts: dict[tuple[Any, Any], _GraphLayout] = {}
        self._layout_pass: _LayoutPass | None = None
        self._iteration: dict[str, Any] | None = None
        self._rank = -1
        self._forward_id = -1
        self._profiled_forwards = 0
        self._prefill_seen = False
        self._forward_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._pending_forward: _PendingForward | None = None
        self._drained_forwards: list[_DrainedForward] = []
        self._layout_slot_counts: dict[tuple[Any, Any], int] = {}

    @property
    def iteration_active(self) -> bool:
        return self._iteration is not None

    @property
    def capture_active(self) -> bool:
        """Whether a layout-discovery or CUDA-graph capture pass is active."""
        return self._layout_pass is not None

    @staticmethod
    def _layout_key(
        cudagraph_mode: Any, batch_descriptor: Any
    ) -> tuple[Any, Any]:
        if getattr(cudagraph_mode, "name", None) == "NONE":
            raise ValueError("CCL profiling requires a captured CUDA graph")
        try:
            hash(cudagraph_mode)
            hash(batch_descriptor)
        except TypeError as error:
            raise TypeError(
                "CUDA-graph mode and batch descriptor must be hashable"
            ) from error
        return cudagraph_mode, batch_descriptor

    def clear_layouts(self) -> None:
        if self.capture_active or self.iteration_active:
            raise RuntimeError("Cannot clear CCL layouts while profiling is active")
        self._layouts.clear()
        self._forward_events = None

    def validate_layouts(
        self, *, require_moe: bool, expected_moe_calls: int
    ) -> None:
        if not self._layouts or any(
            not layout.captured for layout in self._layouts.values()
        ):
            raise RuntimeError("CCL CUDA graph layouts are incomplete")
        if require_moe and any(
            not layout.layers for layout in self._layouts.values()
        ):
            raise RuntimeError(
                "The model's MoE implementation bypasses CCL profiling hooks"
            )
        if expected_moe_calls and any(
            len(layout.layers) != expected_moe_calls
            for layout in self._layouts.values()
        ):
            raise RuntimeError(
                "CCL layouts do not cover every registered MoE layer"
            )

    @contextmanager
    def prepare_layout(self, cudagraph_mode: Any, batch_descriptor: Any):
        """Discover one graph layout during an ordinary dummy pass."""
        if self.capture_active:
            raise RuntimeError("Nested CCL layout passes are not supported")
        if self.iteration_active:
            raise RuntimeError("Cannot prepare a layout during an iteration")

        key = self._layout_key(cudagraph_mode, batch_descriptor)
        layout = _GraphLayout(cudagraph_mode, batch_descriptor)
        state = _LayoutPass("prepare", layout)
        self._layouts[key] = layout
        self._layout_pass = state
        succeeded = False
        try:
            yield
            self._validate_layout_pass(state)
            route_layers = [layer for layer in layout.layers if layer.route_specs]
            route_specs = [layer.route_specs[-1] for layer in route_layers]
            if route_specs:
                devices = {spec.device for spec in route_specs}
                dtypes = {spec.dtype for spec in route_specs}
                if len(devices) != 1 or len(dtypes) != 1:
                    raise RuntimeError(
                        "A CCL layout requires one route device and dtype"
                    )
                device = next(iter(devices))
                if device.type != "cuda":
                    raise RuntimeError("MoE route capture requires a CUDA tensor")
                dtype = next(iter(dtypes))
                route_elements = sum(prod(spec.shape) for spec in route_specs)
                layout.route_buffer = torch.empty(
                    route_elements,
                    dtype=dtype,
                    device=device,
                )
                offset = 0
                for layer, spec in zip(route_layers, route_specs):
                    elements = prod(spec.shape)
                    layer.route_offset = offset
                    layer.route_buffer = layout.route_buffer[
                        offset : offset + elements
                    ].view(spec.shape)
                    offset += elements
            for collective in layout.collectives:
                call_id = collective.metadata.get("layer_call_id")
                if call_id is not None and layout.layers[call_id].route_buffer is None:
                    raise RuntimeError(
                        f"MoE layer {call_id} did not expose routing decisions"
                    )
            if self._forward_events is None:
                self._forward_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for event in self._forward_events:
                    event.record()
            succeeded = True
        finally:
            self._layout_pass = None
            if not succeeded and self._layouts.get(key) is layout:
                del self._layouts[key]

    @contextmanager
    def capture_layout(self, cudagraph_mode: Any, batch_descriptor: Any):
        """Record prepared timing and route nodes into one CUDA graph layout."""
        if self.capture_active:
            raise RuntimeError("Nested CCL layout passes are not supported")
        if self.iteration_active:
            raise RuntimeError("Cannot capture a layout during an iteration")

        key = self._layout_key(cudagraph_mode, batch_descriptor)
        if key not in self._layouts:
            raise RuntimeError(f"CCL layout was not prepared for {key!r}")
        layout = self._layouts[key]
        if layout.captured:
            raise RuntimeError(f"CCL layout was already captured for {key!r}")

        state = _LayoutPass("capture", layout)
        self._layout_pass = state
        try:
            yield
            self._validate_layout_pass(state)
            layout.captured = True
        finally:
            self._layout_pass = None

    @staticmethod
    def _validate_layout_pass(state: _LayoutPass) -> None:
        if state.current_layer_call_id is not None:
            raise RuntimeError("CCL layout ended inside an MoE layer")
        if state.current_ep_direction is not None:
            raise RuntimeError("CCL layout ended inside an EP collective")
        if state.deferred_moe_all_reduce is not None:
            raise RuntimeError("Deferred MoE AllReduce was not recorded")
        if state.open_collective is not None:
            raise RuntimeError("CCL layout has an unfinished collective")
        if state.kind != "capture":
            return

        layout = state.layout
        if state.layer_index != len(layout.layers):
            raise RuntimeError(
                "Captured MoE layer count does not match the prepared layout"
            )
        if state.collective_index != len(layout.collectives):
            raise RuntimeError(
                "Captured collective count does not match the prepared layout"
            )
        for layer in layout.layers:
            if state.route_indices.get(layer.call_id, 0) != len(layer.route_specs):
                raise RuntimeError(
                    "Captured MoE route count does not match the prepared layout"
                )

    def start_iteration(
        self,
        *,
        phase: ProfilePhase,
        rank: int,
        batch_size: int,
        input_len: int,
        output_len: int,
    ) -> None:
        if self.iteration_active:
            raise RuntimeError("A CCL profiling iteration is already active")
        if self.capture_active:
            raise RuntimeError("Cannot start an iteration during graph capture")
        if phase not in ("prefill", "decode"):
            raise ValueError(f"Unknown CCL profiling phase: {phase}")
        if batch_size <= 0 or input_len <= 0 or output_len <= 0:
            raise ValueError("CCL profiling dimensions must be positive")
        if phase == "prefill" and output_len != 1:
            raise ValueError("Prefill profiling requires output_len=1")

        self._iteration = {
            "phase": phase,
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
        }
        self._rank = int(rank)
        self._forward_id = -1
        self._profiled_forwards = 0
        self._prefill_seen = False
        self._pending_forward = None
        self._drained_forwards.clear()
        self._layout_slot_counts.clear()
        self._prepare_iteration_route_slots()

    def _prepare_iteration_route_slots(self) -> None:
        assert self._iteration is not None
        phase = self._iteration["phase"]
        required_tokens = self._iteration["batch_size"]
        expected_forwards = self._iteration["output_len"] - 1
        if phase == "prefill":
            required_tokens *= self._iteration["input_len"]
            expected_forwards = 1

        prepared = False
        for mode_name in ("FULL", "PIECEWISE"):
            candidates = [
                layout
                for layout in self._layouts.values()
                if layout.captured
                and getattr(layout.cudagraph_mode, "name", None) == mode_name
                and not getattr(layout.batch_descriptor, "has_lora", False)
                and getattr(layout.batch_descriptor, "num_tokens", -1)
                >= required_tokens
            ]
            if not candidates:
                continue
            captured_tokens = min(
                layout.batch_descriptor.num_tokens for layout in candidates
            )
            for layout in candidates:
                if layout.batch_descriptor.num_tokens != captured_tokens:
                    continue
                while len(layout.host_route_slots) < expected_forwards:
                    layout.host_route_slots.append(
                        self._allocate_host_route_slot(layout)
                    )
                prepared = True
        if not prepared:
            raise RuntimeError(
                f"No captured CCL layout covers the {phase} case with "
                f"{required_tokens} tokens"
            )

    def select_forward(self, num_scheduled_tokens: dict[str, int]) -> bool:
        if not self.iteration_active:
            return False
        assert self._iteration is not None
        phase: ProfilePhase = self._iteration["phase"]
        batch_size = self._iteration["batch_size"]
        input_len = self._iteration["input_len"]
        lengths = [int(value) for value in num_scheduled_tokens.values()]
        if not lengths:
            return False
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

    def begin_forward(
        self,
        *,
        cudagraph_mode: Any,
        batch_descriptor: Any,
        num_tokens_unpadded: int,
        logical_source_sizes: list[int] | None = None,
    ) -> int:
        if not self.iteration_active:
            raise RuntimeError("No CCL profiling iteration is active")
        if self._pending_forward is not None:
            raise RuntimeError(
                "The previous profiled graph replay has not been drained"
            )
        key = self._layout_key(cudagraph_mode, batch_descriptor)
        layout = self._layouts.get(key)
        if layout is None or not layout.captured:
            raise RuntimeError(f"No captured CCL layout exists for {key!r}")
        if num_tokens_unpadded <= 0:
            raise ValueError("A profiled forward must contain at least one token")
        all_to_all_group_sizes = {
            len(collective.metadata["group_ranks"])
            for collective in layout.collectives
            if collective.metadata["op"] == "all_to_all"
        }
        if len(all_to_all_group_sizes) > 1:
            raise RuntimeError("A CCL layout has inconsistent AlltoAll groups")
        if logical_source_sizes is None:
            group_size = next(iter(all_to_all_group_sizes), 0)
            logical_source_sizes = [int(num_tokens_unpadded)] * group_size
        else:
            logical_source_sizes = [int(size) for size in logical_source_sizes]
        if any(size < 0 for size in logical_source_sizes):
            raise ValueError("Logical EP source sizes must be non-negative")
        for collective in layout.collectives:
            metadata = collective.metadata
            if metadata["op"] == "all_to_all" and len(
                logical_source_sizes
            ) != len(metadata["group_ranks"]):
                raise RuntimeError(
                    "Runtime logical EP sizes do not match the captured group"
                )

        slot_index = self._layout_slot_counts.get(key, 0)
        self._layout_slot_counts[key] = slot_index + 1
        if slot_index >= len(layout.host_route_slots):
            raise RuntimeError(
                "CCL route slots were not prepared before the measured forward"
            )
        host_route_slot = layout.host_route_slots[slot_index]

        if self._forward_events is None:
            raise RuntimeError("CCL forward events were not prepared")
        start, end = self._forward_events
        self._forward_id += 1
        self._profiled_forwards += 1
        host_start_ns = monotonic_ns()
        start.record()
        token = self._forward_id
        self._pending_forward = _PendingForward(
            token=token,
            layout=layout,
            start=start,
            end=end,
            num_tokens_unpadded=int(num_tokens_unpadded),
            logical_source_sizes=logical_source_sizes,
            host_route_buffer=host_route_slot.buffer,
            host_routes=host_route_slot.routes,
            host_start_ns=host_start_ns,
        )
        return token

    def _allocate_host_route_slot(
        self, layout: _GraphLayout
    ) -> _HostRouteSlot:
        if layout.route_buffer is None:
            return _HostRouteSlot(None, {})
        host_buffer = torch.empty(
            layout.route_buffer.shape,
            dtype=layout.route_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )
        layer_ids = {
            int(collective.metadata["layer_call_id"])
            for collective in layout.collectives
            if "layer_call_id" in collective.metadata
        }
        routes: dict[int, torch.Tensor] = {}
        for call_id in layer_ids:
            layer = layout.layers[call_id]
            route_buffer = layer.route_buffer
            if route_buffer is None or layer.route_offset is None:
                raise RuntimeError(
                    f"MoE layer {call_id} has no captured route buffer"
                )
            elements = route_buffer.numel()
            routes[call_id] = host_buffer[
                layer.route_offset : layer.route_offset + elements
            ].view(route_buffer.shape)
        return _HostRouteSlot(host_buffer, routes)

    def end_forward(self, token: int) -> None:
        pending = self._require_pending_forward(token)
        if pending.ended:
            raise RuntimeError("Profiled forward was already ended")
        pending.end.record()
        pending.ended = True

    def enqueue_route_snapshot(self, token: int) -> None:
        """Enqueue route D2H copies before the caller's existing sync."""
        pending = self._require_pending_forward(token)
        if not pending.ended:
            raise RuntimeError("Cannot snapshot routes before ending the forward")
        if pending.routes_enqueued:
            raise RuntimeError("MoE routes were already enqueued")
        if pending.host_route_buffer is not None:
            route_buffer = pending.layout.route_buffer
            assert route_buffer is not None
            pending.host_route_buffer.copy_(route_buffer, non_blocking=True)
        pending.routes_enqueued = True

    def drain_forward(self, token: int) -> None:
        """Resolve one replay after the caller's normal scheduler sync."""
        pending = self._require_pending_forward(token)
        if not pending.ended:
            raise RuntimeError("Cannot drain an unfinished profiled forward")
        if not pending.routes_enqueued:
            raise RuntimeError("MoE route snapshots were not enqueued")

        elapsed_ms = pending.start.elapsed_time(pending.end)
        collective_times = [
            (
                pending.start.elapsed_time(collective.start),
                pending.start.elapsed_time(collective.end),
            )
            for collective in pending.layout.collectives
        ]
        for start_ms, end_ms in collective_times:
            if (
                start_ms < 0
                or end_ms < start_ms
                or end_ms > elapsed_ms + 0.001
            ):
                raise RuntimeError(
                    "CCL graph events are outside their model-forward interval"
                )
        self._drained_forwards.append(
            _DrainedForward(
                forward_id=pending.token,
                layout=pending.layout,
                elapsed_ms=elapsed_ms,
                collective_times=collective_times,
                num_tokens_unpadded=pending.num_tokens_unpadded,
                logical_source_sizes=pending.logical_source_sizes,
                host_routes=pending.host_routes,
                host_start_ns=pending.host_start_ns,
            )
        )
        self._pending_forward = None

    def _require_pending_forward(self, token: int) -> _PendingForward:
        pending = self._pending_forward
        if pending is None or pending.token != token:
            raise RuntimeError(f"Unknown profiled forward token: {token}")
        return pending

    def begin_collective(
        self, metadata: dict[str, Any] | None
    ) -> int | None:
        state = self._layout_pass
        if state is None or metadata is None:
            return None
        if state.open_collective is not None:
            raise RuntimeError("Nested logical collectives are not supported")

        index = state.collective_index
        if state.kind == "prepare":
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            state.layout.collectives.append(
                _CollectiveLayout(dict(metadata), start, end)
            )
        else:
            if not torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "A profiled collective is outside CUDA graph capture"
                )
            if index >= len(state.layout.collectives):
                raise RuntimeError(
                    "Captured more collectives than the prepared CCL layout"
                )
            collective = state.layout.collectives[index]
            if collective.metadata != metadata:
                raise RuntimeError(
                    "Captured collective metadata does not match the prepared layout"
                )
            start, end = collective.start, collective.end

        start.record()
        state.open_collective = index
        state.collective_index += 1
        return index

    def end_collective(self, token: int | None) -> None:
        if token is None:
            return
        state = self._layout_pass
        if state is None or state.open_collective != token:
            raise RuntimeError("CCL collective token does not match the active layout")
        state.layout.collectives[token].end.record()
        state.open_collective = None

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
        state = self._layout_pass
        if state is None:
            yield
            return
        if state.current_layer_call_id is not None:
            raise RuntimeError("Nested MoE profiler layers are not supported")
        if state.deferred_moe_all_reduce is not None:
            raise RuntimeError("A deferred MoE AllReduce was not consumed")

        values = {
            "ep_size": int(ep_size),
            "num_experts": int(num_experts),
            "placement": placement,
            "activation_width": int(activation_width),
            "activation_element_size": int(activation_element_size),
            "has_shared_experts": bool(has_shared_experts),
            "ep_group_ranks": [int(rank) for rank in ep_group_ranks],
        }
        call_id = state.layer_index
        if state.kind == "prepare":
            layer = _MoELayerCall(call_id=call_id, **values)
            state.layout.layers.append(layer)
        else:
            if call_id >= len(state.layout.layers):
                raise RuntimeError(
                    "Captured more MoE calls than the prepared CCL layout"
                )
            layer = state.layout.layers[call_id]
            prepared = {
                "ep_size": layer.ep_size,
                "num_experts": layer.num_experts,
                "placement": layer.placement,
                "activation_width": layer.activation_width,
                "activation_element_size": layer.activation_element_size,
                "has_shared_experts": layer.has_shared_experts,
                "ep_group_ranks": layer.ep_group_ranks,
            }
            if prepared != values:
                raise RuntimeError(
                    "Captured MoE metadata does not match the prepared layout"
                )

        state.layer_index += 1
        state.current_layer_call_id = call_id
        try:
            yield
        finally:
            state.current_layer_call_id = None

    def capture_moe_routes(self, topk_ids: torch.Tensor) -> None:
        state = self._layout_pass
        if state is None:
            return
        call_id = state.current_layer_call_id
        if call_id is None:
            raise RuntimeError("MoE routes were captured outside an MoE layer")
        layer = state.layout.layers[call_id]
        route_index = state.route_indices.get(call_id, 0)
        spec = _RouteSpec(tuple(topk_ids.shape), topk_ids.dtype, topk_ids.device)
        if state.kind == "prepare":
            layer.route_specs.append(spec)
        else:
            if not torch.cuda.is_current_stream_capturing():
                raise RuntimeError("MoE routes are outside CUDA graph capture")
            if route_index >= len(layer.route_specs):
                raise RuntimeError(
                    "Captured more MoE route tensors than the prepared layout"
                )
            if layer.route_specs[route_index] != spec:
                raise RuntimeError(
                    "Captured MoE route shape does not match the prepared layout"
                )
            if route_index == len(layer.route_specs) - 1:
                if layer.route_buffer is None:
                    raise RuntimeError("MoE route buffer was not prepared")
                layer.route_buffer.copy_(topk_ids)
        state.route_indices[call_id] = route_index + 1

    @contextmanager
    def ep_collective(
        self,
        direction: EPDirection,
        logical_source_sizes: list[int] | None = None,
    ):
        state = self._layout_pass
        if state is None:
            yield
            return
        if state.current_ep_direction is not None:
            raise RuntimeError("Nested EP profiler regions are not supported")
        state.current_ep_direction = direction
        state.current_ep_logical_source_sizes = (
            [int(size) for size in logical_source_sizes]
            if logical_source_sizes is not None
            else None
        )
        try:
            yield
        finally:
            state.current_ep_direction = None
            state.current_ep_logical_source_sizes = None

    def defer_moe_final_all_reduce(self, logical_width: int) -> None:
        state = self._layout_pass
        if state is None:
            return
        call_id = state.current_layer_call_id
        if call_id is None:
            raise RuntimeError("Deferred MoE AllReduce has no active layer")
        if state.deferred_moe_all_reduce is not None:
            raise RuntimeError("A deferred MoE AllReduce is already pending")
        state.deferred_moe_all_reduce = (call_id, int(logical_width))

    def all_reduce_metadata(
        self,
        tensor: torch.Tensor,
        *,
        group_ranks: list[int],
        rank_in_group: int,
    ) -> dict[str, Any] | None:
        state = self._layout_pass
        if state is None:
            return None
        metadata: dict[str, Any] = {
            "op": "all_reduce",
            "group_ranks": list(group_ranks),
            "rank_in_group": int(rank_in_group),
        }
        if state.deferred_moe_all_reduce is None:
            metadata["input_bytes"] = int(tensor.numel() * tensor.element_size())
            return metadata

        call_id, logical_width = state.deferred_moe_all_reduce
        state.deferred_moe_all_reduce = None
        metadata.update(
            {
                "layer_call_id": call_id,
                "logical_width": logical_width,
                "tensor_rows": prod(tensor.shape[:-1]),
                "element_size": tensor.element_size(),
            }
        )
        return metadata

    def ep_metadata(
        self,
        *,
        sizes: list[int],
        group_ranks: list[int],
        rank_in_group: int,
    ) -> dict[str, Any] | None:
        state = self._layout_pass
        if (
            state is None
            or state.current_ep_direction is None
            or state.current_layer_call_id is None
        ):
            return None
        sizes = [int(size) for size in sizes]
        group_ranks = [int(rank) for rank in group_ranks]
        if len(sizes) != len(group_ranks):
            raise AssertionError("EP size metadata does not match communicator ranks")
        logical_sizes = state.current_ep_logical_source_sizes or sizes
        if len(logical_sizes) != len(group_ranks):
            raise AssertionError("Logical EP sizes do not match communicator ranks")

        layer = state.layout.layers[state.current_layer_call_id]
        if state.current_ep_direction == "dispatch":
            if state.kind == "prepare":
                layer.route_source_sizes = list(sizes)
                layer.route_group_ranks = list(group_ranks)
            elif (
                layer.route_source_sizes != sizes
                or layer.route_group_ranks != group_ranks
            ):
                raise RuntimeError(
                    "Captured EP dispatch sizes do not match the prepared layout"
                )

        return {
            "op": "all_to_all",
            "direction": state.current_ep_direction,
            "group_ranks": group_ranks,
            "rank_in_group": int(rank_in_group),
            "layer_call_id": layer.call_id,
            "source_sizes": sizes,
            "logical_source_sizes": list(logical_sizes),
        }

    def finish_iteration(self) -> dict[str, Any]:
        if not self.iteration_active:
            raise RuntimeError("No CCL profiling iteration is active")
        if self._pending_forward is not None:
            raise RuntimeError("Cannot finish with an undrained profiled forward")
        self._validate_forward_count()

        records = [
            record
            for forward in self._drained_forwards
            for record in self._resolve_forward(forward)
        ]
        result = {"rank": self._rank, "records": records}

        self._iteration = None
        self._drained_forwards.clear()
        self._layout_slot_counts.clear()
        return result

    def _validate_forward_count(self) -> None:
        assert self._iteration is not None
        phase = self._iteration["phase"]
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

    def _resolve_forward(self, forward: _DrainedForward) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = [
            {
                "op": "forward",
                "forward_id": forward.forward_id,
                "elapsed_ms": forward.elapsed_ms,
                "host_start_ns": forward.host_start_ns,
            }
        ]
        for collective, (start_ms, end_ms) in zip(
            forward.layout.collectives, forward.collective_times
        ):
            metadata = dict(collective.metadata)
            if metadata["op"] == "all_to_all":
                metadata["logical_source_sizes"] = list(
                    forward.logical_source_sizes
                )
            result = {
                "op": metadata["op"],
                "forward_id": forward.forward_id,
                "group_ranks": metadata["group_ranks"],
                "start_offset_ms": start_ms,
                "end_offset_ms": end_ms,
            }
            if metadata["rank_in_group"] == 0:
                if metadata["op"] == "all_reduce":
                    pattern = self._resolve_all_reduce(
                        metadata,
                        forward.host_routes,
                        forward,
                    )
                else:
                    assert metadata["op"] == "all_to_all"
                    pattern = self._resolve_ep(
                        metadata,
                        forward.host_routes,
                        forward.layout,
                    )
                bytes_per_row, source_masks, destination_masks = pattern
                result["bytes_per_row"] = bytes_per_row
                if metadata["op"] == "all_to_all":
                    if metadata["direction"] == "dispatch":
                        physical_source_rows = metadata["source_sizes"]
                    else:
                        rows_per_rank = sum(metadata["source_sizes"])
                        physical_source_rows = [rows_per_rank] * len(
                            metadata["group_ranks"]
                        )
                else:
                    physical_rows = int(metadata.get("tensor_rows", 1))
                    physical_source_rows = [physical_rows] * len(
                        metadata["group_ranks"]
                    )
                result["physical_source_rows"] = physical_source_rows
                result["source_masks"] = base64.b64encode(source_masks).decode(
                    "ascii"
                )
                if destination_masks is not None:
                    result["destination_masks"] = base64.b64encode(
                        destination_masks
                    ).decode("ascii")
            records.append(result)
        return records

    def _resolve_all_reduce(
        self,
        record: dict[str, Any],
        routes_by_layer: dict[int, torch.Tensor],
        forward: _DrainedForward,
    ) -> tuple[int, bytes, bytes | None]:
        ranks = record["group_ranks"]
        full_mask = (1 << len(ranks)) - 1
        if "logical_width" not in record:
            source_masks, destination_masks = _packed_pattern(
                [full_mask], [full_mask], len(ranks)
            )
            return record["input_bytes"], source_masks, destination_masks

        call_id = record["layer_call_id"]
        layer = forward.layout.layers[call_id]
        routes = routes_by_layer.get(call_id)
        if routes is None or routes.ndim != 2:
            raise RuntimeError("MoE routes were not captured for final AllReduce")

        tensor_rows = record["tensor_rows"]
        if (
            layer.route_source_sizes is not None
            and layer.route_group_ranks is not None
        ):
            if self._rank not in layer.route_group_ranks:
                raise RuntimeError("MoE route ranks do not include this rank")
            source_rank = layer.route_group_ranks.index(self._rank)
            route_start = sum(layer.route_source_sizes[:source_rank])
            if len(forward.logical_source_sizes) != len(layer.route_group_ranks):
                raise RuntimeError(
                    "Runtime logical EP sizes do not match MoE route ranks"
                )
            route_count = forward.logical_source_sizes[source_rank]
            if (
                route_count > tensor_rows
                or routes.shape[0] < route_start + route_count
            ):
                raise RuntimeError("MoE routes do not match final AllReduce rows")
            routes = routes[route_start : route_start + route_count]
        else:
            route_count = forward.num_tokens_unpadded
            if route_count > tensor_rows or routes.shape[0] < route_count:
                raise RuntimeError("MoE routes do not match final AllReduce rows")
            routes = routes[:route_count]

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
        self,
        record: dict[str, Any],
        routes_by_layer: dict[int, torch.Tensor],
        layout: _GraphLayout,
    ) -> tuple[int, bytes, bytes | None]:
        call_id = record["layer_call_id"]
        layer = layout.layers[call_id]
        routes = routes_by_layer.get(call_id)
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
