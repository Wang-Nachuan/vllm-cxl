# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import ctypes
import threading
from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ProfilePhase = Literal["prefill", "decode"]
EPDirection = Literal["dispatch", "combine"]

# While CUDA graphs are prepared, CUPTI attaches the innermost pushed external
# ID to every traced CUDA API call, and every GPU record carries the correlation
# ID of the call that launched it; that maps each collective to its NCCL
# kernels. Measured steps trace GPU records only, because API tracing costs host
# time on every CUDA call and stretches host-bound steps such as eager prefill.
# A marker kernel on the compute stream brackets each measured step instead.
# One buffer holds every iteration of a case; see collect_iterations().
_TRACE_BUFFER_BYTES = 256 << 20
_STEP_MARKER = "spin_kernel"


class GpuActivity(NamedTuple):
    correlation_id: int
    start: int
    end: int
    graph_id: int
    stream_id: int
    # Empty for memcpy and memset records.
    name: str


class CorrelationTag(NamedTuple):
    external_id: int
    correlation_id: int


def _is_nccl_kernel(name: str) -> bool:
    return "nccl" in name


class _CuptiTrace:
    """Process-wide CUPTI activity collection.

    CUPTI timestamps every kernel, memcpy, and memset, including nodes replayed
    from CUDA graphs, without adding work to the graph. Records are converted
    only when CUPTI completes a buffer. A large buffer that is flushed and
    replaced between iterations keeps that host work outside profiled steps.
    """

    def __init__(self) -> None:
        self._cupti: Any = None
        self._lock = threading.Lock()
        self._activities: list[GpuActivity] = []
        self._tags: list[CorrelationTag] = []
        self._flushing = False
        # Buffers that filled up on their own instead of being flushed.
        self.overflows = 0
        self.active = False
        self._api_tracing = False

    def initialize(self, hw_trace: bool) -> None:
        """Register CUPTI buffers; call before the CUDA context exists."""
        from cupti import cupti

        if hw_trace:
            from cuda.bindings import driver

            # Blackwell hardware event tracing timestamps kernels without
            # instrumenting them. It needs an initialized driver but no context
            # yet, and it loads libnvperf_host.so, which the pip CUPTI package
            # installs next to libcupti but outside the loader's search path.
            (result,) = driver.cuInit(0)
            if result != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuInit failed: {result}")
            cupti.get_timestamp()
            cupti_path = next(
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libcupti.so" in line
            )
            ctypes.CDLL(
                str(Path(cupti_path).with_name("libnvperf_host.so")),
                mode=ctypes.RTLD_GLOBAL,
            )
            cupti.activity_enable_hw_trace(1)
        # Share one record buffer across threads. Each new buffer stalls the
        # CUDA call that requests it for several milliseconds, and per-thread
        # buffers would make every CUDA-calling thread do so after each flush.
        per_thread = ctypes.c_uint8(0)
        size = ctypes.c_size_t(ctypes.sizeof(per_thread))
        cupti.activity_set_attribute(
            cupti.ActivityAttribute.ATTR_PER_THREAD_BUFFER,
            ctypes.addressof(size),
            ctypes.addressof(per_thread),
        )
        cupti.activity_register_callbacks(
            self._buffer_requested, self._buffer_completed
        )
        kind = cupti.ActivityKind
        self._kernel_kind = int(kind.CONCURRENT_KERNEL)
        self._copy_kinds = (int(kind.MEMCPY), int(kind.MEMSET))
        self._correlation_kind = int(kind.EXTERNAL_CORRELATION)
        self._external_kind = int(cupti.ExternalCorrelationKind.CUSTOM0)
        self._gpu_kinds = (kind.CONCURRENT_KERNEL, kind.MEMCPY, kind.MEMSET)
        # CUPTI emits external correlation records only for traced runtime and
        # driver API calls.
        self._api_kinds = (kind.RUNTIME, kind.DRIVER, kind.EXTERNAL_CORRELATION)
        self._cupti = cupti

    def start(self) -> None:
        """Start recording; call before CUDA graphs are instantiated."""
        if self._cupti is None:
            raise RuntimeError(
                "CUPTI tracing was not initialized before CUDA context creation"
            )
        for kind in self._gpu_kinds + self._api_kinds:
            self._cupti.activity_enable(kind)
        self.active = True
        self._api_tracing = True

    def stop_api_tracing(self) -> None:
        for kind in self._api_kinds:
            self._cupti.activity_disable(kind)
        self._api_tracing = False

    def push(self, external_id: int) -> None:
        self._cupti.activity_push_external_correlation_id(
            self._external_kind, external_id
        )

    def pop(self) -> int:
        return self._cupti.activity_pop_external_correlation_id(self._external_kind)

    @staticmethod
    def mark() -> None:
        torch.cuda._sleep(0)

    def drain(self) -> tuple[list[GpuActivity], list[CorrelationTag]]:
        """Return and clear the records of all GPU work issued so far."""
        torch.cuda.synchronize()
        self._flushing = True
        try:
            self._cupti.activity_flush_all(
                int(self._cupti.ActivityFlag.FLUSH_FORCED)
            )
        finally:
            self._flushing = False
        with self._lock:
            activities, self._activities = self._activities, []
            tags, self._tags = self._tags, []
        # A traced CUDA call makes CUPTI allocate the next buffer here rather
        # than in the first step that follows, which would stall it for
        # milliseconds. Measured steps trace no API calls, so trace this one.
        runtime = self._cupti.ActivityKind.RUNTIME
        if not self._api_tracing:
            self._cupti.activity_enable(runtime)
        torch.cuda.current_stream().query()
        if not self._api_tracing:
            self._cupti.activity_disable(runtime)
        return activities, tags

    @staticmethod
    def _buffer_requested() -> tuple[int, int]:
        return _TRACE_BUFFER_BYTES, 0

    def _buffer_completed(self, records: list[Any]) -> None:
        if not self._flushing:
            self.overflows += 1
        activities = []
        tags = []
        for record in records:
            kind = record.kind
            if kind == self._kernel_kind:
                activities.append(
                    GpuActivity(
                        record.correlation_id,
                        record.start,
                        record.end,
                        record.graph_id,
                        record.stream_id,
                        record.name,
                    )
                )
            elif kind in self._copy_kinds:
                activities.append(
                    GpuActivity(
                        record.correlation_id,
                        record.start,
                        record.end,
                        record.graph_id,
                        record.stream_id,
                        "",
                    )
                )
            elif (
                kind == self._correlation_kind
                and record.external_kind == self._external_kind
            ):
                tags.append(
                    CorrelationTag(record.external_id, record.correlation_id)
                )
        with self._lock:
            self._activities.extend(activities)
            self._tags.extend(tags)


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
    route_tensor: torch.Tensor | None = None
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
    kernel_names: list[str] | None = None


@dataclass
class _GraphLayout:
    cudagraph_mode: Any
    batch_descriptor: Any
    collectives: list[_CollectiveLayout] = field(default_factory=list)
    layers: list[_MoELayerCall] = field(default_factory=list)
    route_buffer: torch.Tensor | None = None
    route_sources: list[torch.Tensor] = field(default_factory=list)
    all_to_all_group_size: int = 0
    sequence_parallel_size: int | None = None
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
    collective_tag_ids: list[int] = field(default_factory=list)


@dataclass
class _ProfiledStep:
    token: int
    layout: _GraphLayout
    num_tokens_unpadded: int
    logical_source_sizes: list[int]
    host_route_buffer: torch.Tensor | None
    host_routes: dict[int, torch.Tensor]
    eager_collectives: list[dict[str, Any]] = field(default_factory=list)


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
    ):
        raise ValueError("MoE routes and EP size metadata do not match")

    if ep_size == 1:
        # Without EP, DP ranks still dispatch and combine, but every rank
        # holds a slice of every expert.
        expert_bits = [(1 << len(collective_group_ranks)) - 1] * num_experts
    else:
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
    """CUDA-graph-aware profiler for logical NCCL collectives.

    An ordinary dummy pass discovers each graph layout's logical collectives,
    the NCCL kernels each one launches, and the MoE route tensors. The
    following CUDA-graph capture inserts nothing into the graph; it only keeps
    references to the captured route tensors, whose memory every replay
    rewrites. A profiled step spans the model forward, logits, and sampling,
    bracketed by two marker kernels. After the iteration ends, the CUPTI
    records between the markers resolve the step's GPU interval and the
    interval of every logical collective.
    """

    def __init__(self) -> None:
        self._trace = _CuptiTrace()
        self._layouts: dict[tuple[Any, Any], _GraphLayout] = {}
        self._layout_pass: _LayoutPass | None = None
        self._iteration: dict[str, Any] | None = None
        self._rank = -1
        self._forward_id = -1
        self._profiled_forwards = 0
        self._prefill_seen = False
        self._step: _ProfiledStep | None = None
        self._finished_steps: list[_ProfiledStep] = []
        self._pending_iterations: list[tuple[int, list[_ProfiledStep]]] = []
        self._layout_slot_counts: dict[tuple[Any, Any], int] = {}
        self._next_tag_id = 1

    @property
    def iteration_active(self) -> bool:
        return self._iteration is not None

    @property
    def capture_active(self) -> bool:
        """Whether a layout-discovery or CUDA-graph capture pass is active."""
        return self._layout_pass is not None

    @property
    def step_active(self) -> bool:
        """Whether a profiled step has started and not finished sampling."""
        return self._step is not None

    def initialize_tracing(self, *, hw_trace: bool) -> None:
        self._trace.initialize(hw_trace)
        logger.info(
            "CCL profiler timestamps GPU kernels with CUPTI %s tracing",
            "hardware" if hw_trace else "software",
        )

    def start_tracing(self) -> None:
        self._trace.start()

    def _new_tag_id(self) -> int:
        tag_id = self._next_tag_id
        self._next_tag_id += 1
        return tag_id

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

    def validate_layouts(
        self, *, require_moe: bool, expected_moe_calls: int
    ) -> None:
        if not self._layouts or any(
            not layout.captured for layout in self._layouts.values()
        ):
            raise RuntimeError("CCL CUDA graph layouts are incomplete")
        if any(
            collective.kernel_names is None
            for layout in self._layouts.values()
            for collective in layout.collectives
        ):
            raise RuntimeError("CCL layouts were prepared without CUPTI tracing")
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
        # Only the layout passes need CUDA API records; measured steps are
        # bracketed by marker kernels. Start measuring with an empty buffer.
        self._trace.stop_api_tracing()
        self._trace.drain()
        self._trace.overflows = 0

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
                layout.route_buffer = torch.empty(
                    sum(prod(spec.shape) for spec in route_specs),
                    dtype=next(iter(dtypes)),
                    device=device,
                )
                offset = 0
                for layer, spec in zip(route_layers, route_specs):
                    layer.route_offset = offset
                    offset += prod(spec.shape)
            for collective in layout.collectives:
                call_id = collective.metadata.get("layer_call_id")
                if call_id is not None and layout.layers[call_id].route_offset is None:
                    raise RuntimeError(
                        f"MoE layer {call_id} did not expose routing decisions"
                    )
            if self._trace.active:
                self._assign_collective_kernels(layout, state.collective_tag_ids)
            succeeded = True
        finally:
            self._layout_pass = None
            if not succeeded and self._layouts.get(key) is layout:
                del self._layouts[key]

    def _assign_collective_kernels(
        self, layout: _GraphLayout, tag_ids: list[int]
    ) -> None:
        """Record the NCCL kernels each collective launched in the dummy pass."""
        activities, tags = self._trace.drain()
        index_of_tag = {tag_id: index for index, tag_id in enumerate(tag_ids)}
        index_of_call = {
            tag.correlation_id: index_of_tag[tag.external_id]
            for tag in tags
            if tag.external_id in index_of_tag
        }
        kernels: dict[int, list[GpuActivity]] = defaultdict(list)
        for activity in activities:
            index = index_of_call.get(activity.correlation_id)
            if index is not None and activity.name:
                kernels[index].append(activity)
        for index, collective in enumerate(layout.collectives):
            names = [
                kernel.name
                for kernel in sorted(kernels[index], key=lambda kernel: kernel.start)
            ]
            if not names or not all(_is_nccl_kernel(name) for name in names):
                raise RuntimeError(
                    f"Logical collective {index} launched unexpected kernels: {names}"
                )
            collective.kernel_names = names

    @contextmanager
    def capture_layout(self, cudagraph_mode: Any, batch_descriptor: Any):
        """Check one CUDA-graph capture against its prepared layout."""
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
            # view() fails rather than silently snapshotting a copy.
            layout.route_sources = [
                layer.route_tensor.view(-1)
                for layer in layout.layers
                if layer.route_tensor is not None
            ]
            group_sizes = {
                len(collective.metadata["group_ranks"])
                for collective in layout.collectives
                if collective.metadata["op"] == "all_to_all"
            }
            if len(group_sizes) > 1:
                raise RuntimeError("A CCL layout has inconsistent AlltoAll groups")
            layout.all_to_all_group_size = next(iter(group_sizes), 0)
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
        if not self._trace.active:
            raise RuntimeError("CUPTI tracing is not active")

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
        self._step = None
        self._finished_steps.clear()
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
        """Start a profiled step; GPU work launched until end_step belongs to it."""
        if not self.iteration_active:
            raise RuntimeError("No CCL profiling iteration is active")
        if self._step is not None:
            raise RuntimeError("The previous profiled step has not finished sampling")
        key = self._layout_key(cudagraph_mode, batch_descriptor)
        layout = self._layouts.get(key)
        if layout is None or not layout.captured:
            raise RuntimeError(f"No captured CCL layout exists for {key!r}")
        if num_tokens_unpadded <= 0:
            raise ValueError("A profiled forward must contain at least one token")
        if logical_source_sizes is None:
            # Static batches give every DP rank the same tokens. Sequence-parallel
            # MoE splits the padded graph batch into one chunk per TP rank, so
            # real tokens fill the leading chunks.
            sp_size = layout.sequence_parallel_size or 1
            chunk = -(-int(layout.batch_descriptor.num_tokens) // sp_size)
            logical_source_sizes = [
                min(max(int(num_tokens_unpadded) - rank * chunk, 0), chunk)
                for rank in range(sp_size)
            ] * (layout.all_to_all_group_size // sp_size)
        else:
            logical_source_sizes = [int(size) for size in logical_source_sizes]
        if any(size < 0 for size in logical_source_sizes):
            raise ValueError("Logical EP source sizes must be non-negative")
        if layout.all_to_all_group_size and (
            len(logical_source_sizes) != layout.all_to_all_group_size
        ):
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

        self._forward_id += 1
        self._profiled_forwards += 1
        step = _ProfiledStep(
            token=self._forward_id,
            layout=layout,
            num_tokens_unpadded=int(num_tokens_unpadded),
            logical_source_sizes=logical_source_sizes,
            host_route_buffer=host_route_slot.buffer,
            host_routes=host_route_slot.routes,
        )
        self._trace.mark()
        self._step = step
        return step.token

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
            if layer.route_offset is None:
                raise RuntimeError(
                    f"MoE layer {call_id} has no captured route tensor"
                )
            shape = layer.route_specs[-1].shape
            routes[call_id] = host_buffer[
                layer.route_offset : layer.route_offset + prod(shape)
            ].view(shape)
        return _HostRouteSlot(host_buffer, routes)

    def end_step(self, token: int) -> None:
        """End a profiled step after sampling and snapshot its MoE routes."""
        step = self._step
        if step is None or step.token != token:
            raise RuntimeError(f"Unknown profiled step token: {token}")
        self._trace.mark()
        if step.host_route_buffer is not None:
            route_buffer = step.layout.route_buffer
            assert route_buffer is not None
            torch.cat(step.layout.route_sources, out=route_buffer)
            step.host_route_buffer.copy_(route_buffer, non_blocking=True)
        self._finished_steps.append(step)
        self._step = None

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
            state.layout.collectives.append(_CollectiveLayout(dict(metadata)))
            if self._trace.active:
                tag_id = self._new_tag_id()
                state.collective_tag_ids.append(tag_id)
                self._trace.push(tag_id)
        else:
            if not torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "A profiled collective is outside CUDA graph capture"
                )
            if index >= len(state.layout.collectives):
                raise RuntimeError(
                    "Captured more collectives than the prepared CCL layout"
                )
            if state.layout.collectives[index].metadata != metadata:
                raise RuntimeError(
                    "Captured collective metadata does not match the prepared layout"
                )

        state.open_collective = index
        state.collective_index += 1
        return index

    def end_collective(self, token: int | None) -> None:
        if token is None:
            return
        state = self._layout_pass
        if state is None or state.open_collective != token:
            raise RuntimeError("CCL collective token does not match the active layout")
        if state.kind == "prepare" and self._trace.active:
            self._trace.pop()
        state.open_collective = None

    def add_eager_collective(self, metadata: dict[str, Any]) -> None:
        """Record an eager collective that the profiled step launches next."""
        if self._step is None:
            raise RuntimeError("Eager CCL collectives require a profiled step")
        self._step.eager_collectives.append(metadata)

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
                # The reference keeps this graph-pool block out of reuse, so
                # every replay leaves the layer's routes in it.
                layer.route_tensor = topk_ids
        state.route_indices[call_id] = route_index + 1

    @contextmanager
    def ep_collective(
        self,
        direction: EPDirection,
        logical_source_sizes: list[int] | None = None,
        sequence_parallel_size: int = 1,
    ):
        state = self._layout_pass
        if state is None:
            yield
            return
        if state.current_ep_direction is not None:
            raise RuntimeError("Nested EP profiler regions are not supported")
        layout = state.layout
        if state.kind == "prepare" and layout.sequence_parallel_size is None:
            layout.sequence_parallel_size = sequence_parallel_size
        elif layout.sequence_parallel_size != sequence_parallel_size:
            raise RuntimeError("EP collectives disagree on sequence parallelism")
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

    @staticmethod
    def value_independent_metadata(
        op: str,
        tensor: torch.Tensor,
        *,
        group_ranks: list[int],
        rank_in_group: int,
    ) -> dict[str, Any]:
        """Describe an AllGather or ReduceScatter by its per-rank input."""
        assert op in ("all_gather", "reduce_scatter")
        return {
            "op": op,
            "group_ranks": [int(rank) for rank in group_ranks],
            "rank_in_group": int(rank_in_group),
            "input_bytes": int(tensor.numel() * tensor.element_size()),
        }

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

    def finish_iteration(self) -> None:
        """Keep the iteration's steps until collect_iterations()."""
        if not self.iteration_active:
            raise RuntimeError("No CCL profiling iteration is active")
        if self._step is not None:
            raise RuntimeError("Cannot finish with an unfinished profiled step")
        self._validate_forward_count()
        self._pending_iterations.append((self._rank, list(self._finished_steps)))
        self._iteration = None
        self._finished_steps.clear()
        self._layout_slot_counts.clear()

    def collect_iterations(self, last: int) -> list[dict[str, Any]]:
        """Resolve the last finished iterations from one CUPTI flush.

        Changing CUPTI state slows the requests that follow it for several
        milliseconds, so the caller flushes once per case and lets warmup
        iterations absorb the cost.
        """
        if self.iteration_active:
            raise RuntimeError("Cannot collect during a profiling iteration")
        pending, self._pending_iterations = self._pending_iterations, []
        if not 0 < last <= len(pending):
            raise ValueError(
                f"Cannot collect the last {last} of {len(pending)} iterations"
            )
        activities, _ = self._trace.drain()
        overflows, self._trace.overflows = self._trace.overflows, 0
        if overflows:
            raise RuntimeError(
                "The CUPTI record buffer filled during profiled iterations, so "
                "its conversion ran during measured steps; increase "
                "_TRACE_BUFFER_BYTES"
            )
        if any(
            activity.start <= 0 or activity.end < activity.start
            for activity in activities
        ):
            raise RuntimeError("CUPTI lost GPU timestamps in a profiled iteration")
        # Each step launched one marker before its forward and one after
        # sampling on its compute stream; its GPU work is what ran between them.
        # Graph replays stay in stream order but report the internal streams
        # of their kernels, so kernels count from any stream. Copies count
        # only from the compute stream, which excludes vLLM's async output
        # copies of earlier steps.
        activities.sort(key=lambda activity: activity.start)
        markers = [
            activity for activity in activities if _STEP_MARKER in activity.name
        ]
        step_count = sum(len(steps) for _, steps in pending)
        if len(markers) != 2 * step_count:
            raise RuntimeError(
                f"CUPTI recorded {len(markers)} step markers for "
                f"{step_count} profiled steps"
            )
        streams = {marker.stream_id for marker in markers}
        if len(streams) != 1:
            raise RuntimeError("Profiled steps ran on different CUDA streams")
        stream_id = streams.pop()
        work = [
            activity
            for activity in activities
            if (activity.name or activity.stream_id == stream_id)
            and _STEP_MARKER not in activity.name
        ]
        work_starts = [activity.start for activity in work]

        results = []
        marker_index = 2 * sum(len(steps) for _, steps in pending[:-last])
        for rank, steps in pending[-last:]:
            records = []
            for step in steps:
                begin, end = markers[marker_index], markers[marker_index + 1]
                marker_index += 2
                window = work[
                    bisect_left(work_starts, begin.end) : bisect_left(
                        work_starts, end.start
                    )
                ]
                records.extend(
                    self._resolve_step(
                        step,
                        [activity for activity in window if activity.end <= end.start],
                    )
                )
            results.append({"rank": rank, "records": records})
        return results

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

    def _resolve_step(
        self,
        step: _ProfiledStep,
        activities: list[GpuActivity],
    ) -> list[dict[str, Any]]:
        if not activities:
            raise RuntimeError(f"CUPTI recorded no GPU work for step {step.token}")
        nccl_kernels = [
            activity for activity in activities if _is_nccl_kernel(activity.name)
        ]

        # A logical collective spans from its first NCCL kernel start to its
        # last NCCL kernel end.
        layout = step.layout
        graph_kernels = [kernel for kernel in nccl_kernels if kernel.graph_id]
        expected_names = [
            name
            for collective in layout.collectives
            for name in collective.kernel_names or ()
        ]
        if [kernel.name for kernel in graph_kernels] != expected_names:
            raise RuntimeError(
                f"Step {step.token} replayed NCCL kernels "
                f"{[kernel.name for kernel in graph_kernels]}, expected "
                f"{expected_names}"
            )
        timings: list[tuple[dict[str, Any], int, int]] = []
        position = 0
        for collective in layout.collectives:
            assert collective.kernel_names is not None
            kernels = graph_kernels[position : position + len(collective.kernel_names)]
            position += len(kernels)
            timings.append(
                (
                    collective.metadata,
                    kernels[0].start,
                    max(kernel.end for kernel in kernels),
                )
            )

        # Each eager collective, such as the logits all-gather, launches one
        # NCCL kernel outside the graphs, in step order.
        eager_kernels = [kernel for kernel in nccl_kernels if not kernel.graph_id]
        if len(eager_kernels) != len(step.eager_collectives):
            raise RuntimeError(
                f"Step {step.token} ran eager NCCL kernels "
                f"{[kernel.name for kernel in eager_kernels]} for "
                f"{len(step.eager_collectives)} recorded eager collectives"
            )
        for metadata, kernel in zip(step.eager_collectives, eager_kernels):
            timings.append((metadata, kernel.start, kernel.end))

        records: list[dict[str, Any]] = [
            {
                "op": "forward",
                "forward_id": step.token,
                "start_ns": activities[0].start,
                "end_ns": max(activity.end for activity in activities),
            }
        ]
        for metadata, start_ns, end_ns in timings:
            metadata = dict(metadata)
            if metadata["op"] == "all_to_all":
                metadata["logical_source_sizes"] = list(step.logical_source_sizes)
            result = {
                "op": metadata["op"],
                "forward_id": step.token,
                "group_ranks": metadata["group_ranks"],
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
            if metadata["rank_in_group"] == 0:
                group_size = len(metadata["group_ranks"])
                if metadata["op"] == "all_reduce":
                    pattern = self._resolve_all_reduce(metadata, step)
                    physical_source_rows = [
                        int(metadata.get("tensor_rows", 1))
                    ] * group_size
                elif metadata["op"] == "all_gather":
                    pattern = (
                        metadata["input_bytes"],
                        *_packed_pattern(
                            [1 << rank for rank in range(group_size)],
                            [(1 << group_size) - 1] * group_size,
                            group_size,
                        ),
                    )
                    physical_source_rows = [1] * group_size
                elif metadata["op"] == "reduce_scatter":
                    # Every rank contributes every chunk; chunk i goes to rank i.
                    pattern = (
                        metadata["input_bytes"] // group_size,
                        *_packed_pattern(
                            [(1 << group_size) - 1] * group_size,
                            [1 << rank for rank in range(group_size)],
                            group_size,
                        ),
                    )
                    physical_source_rows = [group_size] * group_size
                else:
                    assert metadata["op"] == "all_to_all"
                    pattern = self._resolve_ep(metadata, step)
                    if metadata["direction"] == "dispatch":
                        physical_source_rows = metadata["source_sizes"]
                    else:
                        physical_source_rows = [
                            sum(metadata["source_sizes"])
                        ] * group_size
                bytes_per_row, source_masks, destination_masks = pattern
                result["bytes_per_row"] = bytes_per_row
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
        step: _ProfiledStep,
    ) -> tuple[int, bytes, bytes | None]:
        ranks = record["group_ranks"]
        full_mask = (1 << len(ranks)) - 1
        if "logical_width" not in record:
            source_masks, destination_masks = _packed_pattern(
                [full_mask], [full_mask], len(ranks)
            )
            return record["input_bytes"], source_masks, destination_masks

        call_id = record["layer_call_id"]
        layer = step.layout.layers[call_id]
        routes = step.host_routes.get(call_id)
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
            if len(step.logical_source_sizes) != len(layer.route_group_ranks):
                raise RuntimeError(
                    "Runtime logical EP sizes do not match MoE route ranks"
                )
            route_count = step.logical_source_sizes[source_rank]
            if (
                route_count > tensor_rows
                or routes.shape[0] < route_start + route_count
            ):
                raise RuntimeError("MoE routes do not match final AllReduce rows")
            routes = routes[route_start : route_start + route_count]
        else:
            route_count = step.num_tokens_unpadded
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

    @staticmethod
    def _resolve_ep(
        record: dict[str, Any],
        step: _ProfiledStep,
    ) -> tuple[int, bytes, bytes | None]:
        call_id = record["layer_call_id"]
        layer = step.layout.layers[call_id]
        routes = step.host_routes.get(call_id)
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
