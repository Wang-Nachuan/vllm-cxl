# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.ccl_profiler as ccl_profiler


class _Mode(Enum):
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class _BatchDescriptor:
    num_tokens: int
    uniform: bool
    has_lora: bool = False


class _FakeTensor:
    shape = (2, 4)

    @staticmethod
    def numel() -> int:
        return 8

    @staticmethod
    def element_size() -> int:
        return 2


class _FakeTrace:
    """Stands in for CUPTI: external correlation records while API tracing is
    on, and marker kernels on the compute stream."""

    active = True
    overflows = 0

    def __init__(self) -> None:
        self.stack: list[int] = []
        self.pushes = 0
        self.api_tracing = True
        self.activities: list[ccl_profiler.GpuActivity] = []
        self.tags: list[ccl_profiler.CorrelationTag] = []
        self._correlation_id = 0
        self._clock = 0

    def push(self, external_id: int) -> None:
        self.pushes += 1
        self.stack.append(external_id)

    def pop(self) -> int:
        return self.stack.pop()

    def stop_api_tracing(self) -> None:
        self.api_tracing = False

    def launch(
        self,
        kernels: list[tuple[str, int, int]],
        *,
        graph_id: int = 0,
        stream_id: int = 7,
    ) -> None:
        """Record one API call; a graph launch runs several kernels."""
        self._correlation_id += 1
        for name, start, end in kernels:
            self.activities.append(
                ccl_profiler.GpuActivity(
                    self._correlation_id, start, end, graph_id, stream_id, name
                )
            )
            self._clock = max(self._clock, end)
        if self.api_tracing and self.stack:
            self.tags.append(
                ccl_profiler.CorrelationTag(self.stack[-1], self._correlation_id)
            )

    def mark(self) -> None:
        """The marker runs right after the GPU work launched before it."""
        self.launch([(ccl_profiler._STEP_MARKER, self._clock + 1, self._clock + 2)])

    def drain(self):
        drained = (self.activities, self.tags)
        self.activities, self.tags = [], []
        return drained


def _traced_profiler() -> tuple[ccl_profiler.CCLProfiler, _FakeTrace]:
    profiler = ccl_profiler.CCLProfiler()
    trace = _FakeTrace()
    profiler._trace = trace  # type: ignore[assignment]
    return profiler, trace


def _all_reduce(profiler: ccl_profiler.CCLProfiler, kernels, trace) -> None:
    metadata = profiler.all_reduce_metadata(
        _FakeTensor(),  # type: ignore[arg-type]
        group_ranks=[0, 1],
        rank_in_group=0,
    )
    token = profiler.begin_collective(metadata)
    if kernels:
        trace.launch(kernels)
    profiler.end_collective(token)


def _captured_decode_profiler(
    monkeypatch,
) -> tuple[ccl_profiler.CCLProfiler, _FakeTrace]:
    """A decode layout whose second AllReduce launches two NCCL kernels."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    profiler, trace = _traced_profiler()
    descriptor = _BatchDescriptor(num_tokens=1, uniform=True)
    with profiler.prepare_layout(_Mode.FULL, descriptor):
        _all_reduce(profiler, [("ncclDevKernel_AllReduce", 10, 20)], trace)
        trace.launch([("gemm", 30, 40)])
        _all_reduce(
            profiler,
            [("ncclDevKernel_Reduce", 50, 60), ("ncclDevKernel_Reduce", 70, 80)],
            trace,
        )
    with profiler.capture_layout(_Mode.FULL, descriptor):
        _all_reduce(profiler, None, trace)
        _all_reduce(profiler, None, trace)
    profiler.start_iteration(
        phase="decode", rank=0, batch_size=1, input_len=4, output_len=2
    )
    assert not profiler.select_forward({"request": 4})
    assert profiler.select_forward({"request": 1})
    return profiler, trace


def _run_decode_step(
    profiler: ccl_profiler.CCLProfiler,
    trace: _FakeTrace,
    graph_kernels: list[tuple[str, int, int]],
    offset: int = 0,
) -> None:
    token = profiler.begin_forward(
        cudagraph_mode=_Mode.FULL,
        batch_descriptor=_BatchDescriptor(num_tokens=1, uniform=True),
        num_tokens_unpadded=1,
    )
    trace.launch(graph_kernels, graph_id=7)
    metadata = profiler.value_independent_metadata(
        "all_gather",
        _FakeTensor(),  # type: ignore[arg-type]
        group_ranks=[0, 1],
        rank_in_group=0,
    )
    profiler.add_eager_collective(metadata)
    trace.launch([("ncclDevKernel_AllGather", offset + 1500, offset + 1600)])
    trace.launch([("argmax", offset + 1700, offset + 1750)])
    profiler.end_step(token)
    trace.launch([("next_step_input_copy", offset + 2000, offset + 2010)])


def _finish(profiler: ccl_profiler.CCLProfiler) -> dict:
    profiler.finish_iteration()
    return profiler.collect_iterations(1)[0]


def test_ep_patterns_follow_routes_and_direction() -> None:
    routes = torch.tensor([[0, 2], [3, 3], [1, 2]], dtype=torch.int32)

    dispatch = ccl_profiler._ep_pattern(
        routes,
        source_sizes=[2, 1],
        logical_source_sizes=[2, 1],
        direction="dispatch",
        num_experts=4,
        ep_size=2,
        ep_group_ranks=[0, 1],
        collective_group_ranks=[0, 1],
        placement="linear",
        bytes_per_row=16,
    )
    combine = ccl_profiler._ep_pattern(
        routes,
        source_sizes=[2, 1],
        logical_source_sizes=[2, 1],
        direction="combine",
        num_experts=4,
        ep_size=2,
        ep_group_ranks=[0, 1],
        collective_group_ranks=[0, 1],
        placement="linear",
        bytes_per_row=16,
    )

    assert dispatch == (16, b"\x01\x01\x02", b"\x03\x02\x03")
    assert combine == (16, b"\x03\x02\x03", b"\x01\x01\x02")


def test_ep_patterns_without_ep_reach_every_dp_rank() -> None:
    routes = torch.tensor([[0, 2], [3, 3], [1, 2]], dtype=torch.int32)
    kwargs = dict(
        source_sizes=[2, 1],
        logical_source_sizes=[2, 1],
        num_experts=4,
        ep_size=1,
        ep_group_ranks=[0, 1, 2, 3],
        collective_group_ranks=[0, 2],
        placement="linear",
        bytes_per_row=16,
    )

    dispatch = ccl_profiler._ep_pattern(routes, direction="dispatch", **kwargs)
    combine = ccl_profiler._ep_pattern(routes, direction="combine", **kwargs)

    assert dispatch == (16, b"\x01\x01\x02", None)
    assert combine == (16, b"\x03\x03\x03", b"\x01\x01\x02")


def test_round_robin_expert_owners() -> None:
    owners = ccl_profiler._expert_owner_table(7, 3, "round_robin")

    assert owners.tolist() == [0, 1, 2, 0, 1, 2, 0]


def test_masks_use_fixed_width_little_endian_rows() -> None:
    masks = ccl_profiler._pack_masks([1, 1 << 8, (1 << 8) | 1], 9)

    assert masks == b"\x01\x00\x00\x01\x01\x01"


def test_value_independent_all_reduce_uses_one_full_source_mask() -> None:
    profiler = ccl_profiler.CCLProfiler()

    pattern = profiler._resolve_all_reduce(
        {"group_ranks": [0, 1], "input_bytes": 64},
        SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert pattern == (64, b"\x03", None)


def test_ep_pattern_ignores_padding_and_experts_outside_group() -> None:
    routes = torch.tensor([[2], [1], [0]], dtype=torch.int32)

    dispatch = ccl_profiler._ep_pattern(
        routes,
        source_sizes=[2, 1],
        logical_source_sizes=[1, 1],
        direction="dispatch",
        num_experts=4,
        ep_size=4,
        ep_group_ranks=[0, 1, 2, 3],
        collective_group_ranks=[0, 2],
        placement="linear",
        bytes_per_row=8,
    )

    assert dispatch == (8, b"\x01\x02", b"\x02\x01")


def test_moe_all_reduce_uses_runtime_logical_rows() -> None:
    profiler = ccl_profiler.CCLProfiler()
    profiler._rank = 2
    layer = ccl_profiler._MoELayerCall(
        call_id=0,
        ep_size=4,
        num_experts=4,
        placement="linear",
        activation_width=4,
        activation_element_size=2,
        has_shared_experts=False,
        ep_group_ranks=[0, 1, 2, 3],
        route_source_sizes=[2, 1],
        route_group_ranks=[0, 2],
    )
    layout = ccl_profiler._GraphLayout("FULL", "decode", layers=[layer])
    step = ccl_profiler._ProfiledStep(
        token=0,
        layout=layout,
        num_tokens_unpadded=1,
        logical_source_sizes=[1, 1],
        host_route_buffer=None,
        host_routes={0: torch.tensor([[2], [2], [2]], dtype=torch.int32)},
    )
    record = {
        "group_ranks": [2, 3],
        "rank_in_group": 0,
        "layer_call_id": 0,
        "logical_width": 4,
        "tensor_rows": 1,
        "element_size": 2,
    }
    pattern = profiler._resolve_all_reduce(record, step)

    assert pattern == (8, b"\x01", None)


def test_prepare_pass_records_kernels_and_capture_adds_nothing(
    monkeypatch,
) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)
    layout = next(iter(profiler._layouts.values()))

    assert layout.captured
    assert [collective.kernel_names for collective in layout.collectives] == [
        ["ncclDevKernel_AllReduce"],
        ["ncclDevKernel_Reduce", "ncclDevKernel_Reduce"],
    ]
    # Only the two prepare-pass collectives were tagged; capture adds no work.
    assert trace.pushes == 2
    assert trace.stack == []


def test_step_records_use_first_and_last_kernel_of_each_collective(
    monkeypatch,
) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)

    _run_decode_step(
        profiler,
        trace,
        [
            ("embedding", 1000, 1010),
            ("ncclDevKernel_AllReduce", 1020, 1100),
            ("gemm", 1110, 1200),
            ("ncclDevKernel_Reduce", 1210, 1250),
            ("ncclDevKernel_Reduce", 1300, 1400),
        ],
    )
    result = _finish(profiler)

    assert result["rank"] == 0
    forward, first, second, gather = result["records"]
    assert forward == {
        "op": "forward",
        "forward_id": 0,
        "start_ns": 1000,
        "end_ns": 1750,
    }
    assert (first["start_ns"], first["end_ns"]) == (1020, 1100)
    # A logical collective spans its NCCL kernels end to end; the gap between
    # them counts, unlike a sum of kernel durations.
    assert (second["start_ns"], second["end_ns"]) == (1210, 1400)
    assert gather == {
        "op": "all_gather",
        "forward_id": 0,
        "group_ranks": [0, 1],
        "start_ns": 1500,
        "end_ns": 1600,
        "bytes_per_row": 16,
        "physical_source_rows": [1, 1],
        "source_masks": "AQI=",
    }


def test_graph_all_gather_and_reduce_scatter_are_logical_collectives(
    monkeypatch,
) -> None:
    """Sequence-parallel MoE replays TP AllGather/ReduceScatter in the graph."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    profiler, trace = _traced_profiler()
    descriptor = _BatchDescriptor(num_tokens=1, uniform=True)

    def collectives(kernels) -> None:
        for op, kernel in zip(("reduce_scatter", "all_gather"), kernels):
            token = profiler.begin_collective(
                profiler.value_independent_metadata(
                    op,
                    _FakeTensor(),  # type: ignore[arg-type]
                    group_ranks=[0, 1],
                    rank_in_group=0,
                )
            )
            if kernel:
                trace.launch([kernel])
            profiler.end_collective(token)

    with profiler.prepare_layout(_Mode.FULL, descriptor):
        collectives(
            [
                ("ncclDevKernel_ReduceScatter", 10, 20),
                ("ncclDevKernel_AllGather", 30, 40),
            ]
        )
    with profiler.capture_layout(_Mode.FULL, descriptor):
        collectives([None, None])
    profiler.start_iteration(
        phase="decode", rank=0, batch_size=1, input_len=4, output_len=2
    )
    assert not profiler.select_forward({"request": 4})
    assert profiler.select_forward({"request": 1})
    token = profiler.begin_forward(
        cudagraph_mode=_Mode.FULL,
        batch_descriptor=descriptor,
        num_tokens_unpadded=1,
    )
    trace.launch(
        [
            ("ncclDevKernel_ReduceScatter", 1000, 1100),
            ("ncclDevKernel_AllGather", 1200, 1300),
        ],
        graph_id=7,
    )
    profiler.end_step(token)

    _, scatter, gather = _finish(profiler)["records"]

    # The 16-byte input splits into one 8-byte chunk per rank, reduced from
    # both ranks and delivered to its owner.
    assert scatter == {
        "op": "reduce_scatter",
        "forward_id": 0,
        "group_ranks": [0, 1],
        "start_ns": 1000,
        "end_ns": 1100,
        "bytes_per_row": 8,
        "physical_source_rows": [2, 2],
        "source_masks": "AwM=",
        "destination_masks": "AQI=",
    }
    assert (gather["op"], gather["bytes_per_row"]) == ("all_gather", 16)
    assert gather["physical_source_rows"] == [1, 1]
    assert gather["source_masks"] == "AQI="
    assert "destination_masks" not in gather


def test_sequence_parallel_steps_skip_padding_chunks(monkeypatch) -> None:
    profiler, _ = _captured_decode_profiler(monkeypatch)
    layout = next(iter(profiler._layouts.values()))
    # TP2 x DP2 sequence-parallel EP: one decode token per DP rank lands on
    # TP rank 0; TP rank 1 receives only the padding row.
    layout.all_to_all_group_size = 4
    layout.sequence_parallel_size = 2

    profiler.begin_forward(
        cudagraph_mode=_Mode.FULL,
        batch_descriptor=_BatchDescriptor(num_tokens=1, uniform=True),
        num_tokens_unpadded=1,
    )

    assert profiler._step.logical_source_sizes == [1, 0, 1, 0]


def test_collect_resolves_only_the_last_iterations(monkeypatch) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)
    graph_kernels = [
        ("ncclDevKernel_AllReduce", 1020, 1030),
        ("ncclDevKernel_Reduce", 1040, 1050),
        ("ncclDevKernel_Reduce", 1060, 1070),
    ]
    for offset in (0, 2000):
        if offset:
            profiler.start_iteration(
                phase="decode", rank=0, batch_size=1, input_len=4, output_len=2
            )
            assert not profiler.select_forward({"request": 4})
            assert profiler.select_forward({"request": 1})
        kernels = [
            (name, offset + start, offset + end) for name, start, end in graph_kernels
        ]
        _run_decode_step(profiler, trace, kernels, offset)
        profiler.finish_iteration()

    (iteration,) = profiler.collect_iterations(1)
    collectives = iteration["records"][1:3]

    assert [(record["start_ns"], record["end_ns"]) for record in collectives] == [
        (3020, 3030),
        (3040, 3070),
    ]


def test_unexpected_graph_nccl_kernels_fail(monkeypatch) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)

    _run_decode_step(
        profiler,
        trace,
        [
            ("ncclDevKernel_AllReduce", 1020, 1100),
            ("ncclDevKernel_Reduce", 1210, 1250),
        ],
    )

    with pytest.raises(RuntimeError, match="replayed NCCL kernels"):
        _finish(profiler)


def test_unrecorded_eager_nccl_kernel_fails(monkeypatch) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)
    token = profiler.begin_forward(
        cudagraph_mode=_Mode.FULL,
        batch_descriptor=_BatchDescriptor(num_tokens=1, uniform=True),
        num_tokens_unpadded=1,
    )
    trace.launch(
        [
            ("ncclDevKernel_AllReduce", 1020, 1100),
            ("ncclDevKernel_Reduce", 1210, 1250),
            ("ncclDevKernel_Reduce", 1300, 1400),
        ],
        graph_id=7,
    )
    trace.launch([("ncclDevKernel_AllGather", 1500, 1600)])
    profiler.end_step(token)

    with pytest.raises(RuntimeError, match="eager NCCL kernels"):
        _finish(profiler)


def test_steps_keep_graph_kernels_and_compute_stream_copies(monkeypatch) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)
    token = profiler.begin_forward(
        cudagraph_mode=_Mode.FULL,
        batch_descriptor=_BatchDescriptor(num_tokens=1, uniform=True),
        num_tokens_unpadded=1,
    )
    # Graph replays report the internal streams of their kernels.
    trace.launch(
        [
            ("ncclDevKernel_AllReduce", 1020, 1100),
            ("ncclDevKernel_Reduce", 1210, 1250),
            ("ncclDevKernel_Reduce", 1300, 1400),
        ],
        graph_id=7,
        stream_id=645,
    )
    # An async output copy of an earlier step runs on its own copy stream.
    trace.launch([("", 1000, 1800)], stream_id=21)
    trace.launch([("", 1450, 1500)])
    profiler.end_step(token)
    trace.launch([("next_step_input_kernel", 2000, 2010)])

    forward = _finish(profiler)["records"][0]

    assert (forward["start_ns"], forward["end_ns"]) == (1020, 1500)


def test_lost_cupti_timestamps_fail(monkeypatch) -> None:
    profiler, trace = _captured_decode_profiler(monkeypatch)

    _run_decode_step(
        profiler,
        trace,
        [
            ("ncclDevKernel_AllReduce", 0, 0),
            ("ncclDevKernel_Reduce", 1210, 1250),
            ("ncclDevKernel_Reduce", 1300, 1400),
        ],
    )

    with pytest.raises(RuntimeError, match="lost GPU timestamps"):
        _finish(profiler)


def test_route_snapshot_reads_the_held_graph_tensor_after_replay() -> None:
    route = torch.zeros((2, 2), dtype=torch.int32)
    layout = ccl_profiler._GraphLayout(
        "FULL",
        "decode",
        route_buffer=torch.empty(4, dtype=torch.int32),
        route_sources=[route.view(-1)],
    )
    host_buffer = torch.empty(4, dtype=torch.int32)
    profiler, _ = _traced_profiler()
    profiler._step = ccl_profiler._ProfiledStep(
        token=0,
        layout=layout,
        num_tokens_unpadded=2,
        logical_source_sizes=[],
        host_route_buffer=host_buffer,
        host_routes={},
    )

    # A graph replay overwrites the captured tensor in place.
    route.copy_(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    profiler.end_step(0)

    assert host_buffer.tolist() == [1, 2, 3, 4]


def test_decode_preallocation_accepts_piecewise_only_graphs() -> None:
    descriptor = _BatchDescriptor(num_tokens=4, uniform=False)
    layout = ccl_profiler._GraphLayout(
        _Mode.PIECEWISE,
        descriptor,
        captured=True,
    )
    profiler, _ = _traced_profiler()
    profiler._layouts[(_Mode.PIECEWISE, descriptor)] = layout

    profiler.start_iteration(
        phase="decode",
        rank=0,
        batch_size=4,
        input_len=16,
        output_len=2,
    )

    assert len(layout.host_route_slots) == 1


def test_prefill_preallocates_every_possible_graph_mode() -> None:
    full_descriptor = _BatchDescriptor(num_tokens=1, uniform=True)
    piecewise_descriptor = _BatchDescriptor(num_tokens=1, uniform=False)
    full = ccl_profiler._GraphLayout(
        _Mode.FULL,
        full_descriptor,
        captured=True,
    )
    piecewise = ccl_profiler._GraphLayout(
        _Mode.PIECEWISE,
        piecewise_descriptor,
        captured=True,
    )
    profiler, _ = _traced_profiler()
    profiler._layouts = {
        (_Mode.FULL, full_descriptor): full,
        (_Mode.PIECEWISE, piecewise_descriptor): piecewise,
    }

    profiler.start_iteration(
        phase="prefill",
        rank=0,
        batch_size=1,
        input_len=1,
        output_len=1,
    )

    assert len(full.host_route_slots) == 1
    assert len(piecewise.host_route_slots) == 1


def test_zero_token_scheduler_cleanup_is_not_a_profiled_forward() -> None:
    descriptor = _BatchDescriptor(num_tokens=1, uniform=True)
    profiler, _ = _traced_profiler()
    profiler._layouts[(_Mode.FULL, descriptor)] = ccl_profiler._GraphLayout(
        _Mode.FULL,
        descriptor,
        captured=True,
    )
    profiler.start_iteration(
        phase="prefill",
        rank=0,
        batch_size=1,
        input_len=1,
        output_len=1,
    )

    assert not profiler.select_forward({})
    assert profiler.select_forward({"request": 1})
    assert not profiler.select_forward({})


def test_deferred_moe_all_reduce_is_consumed_by_next_collective() -> None:
    class FakeTensor:
        shape = (3, 8)

        @staticmethod
        def numel() -> int:
            return 24

        @staticmethod
        def element_size() -> int:
            return 2

    layer = ccl_profiler._MoELayerCall(
        call_id=0,
        ep_size=2,
        num_experts=4,
        placement="linear",
        activation_width=8,
        activation_element_size=2,
        has_shared_experts=False,
        ep_group_ranks=[0, 1],
    )
    layout = ccl_profiler._GraphLayout("FULL", "decode", layers=[layer])
    state = ccl_profiler._LayoutPass("prepare", layout)
    state.current_layer_call_id = 0
    profiler = ccl_profiler.CCLProfiler()
    profiler._layout_pass = state

    profiler.defer_moe_final_all_reduce(8)
    state.current_layer_call_id = None
    metadata = profiler.all_reduce_metadata(
        FakeTensor(),  # type: ignore[arg-type]
        group_ranks=[0, 1],
        rank_in_group=0,
    )

    assert state.deferred_moe_all_reduce is None
    assert metadata == {
        "op": "all_reduce",
        "group_ranks": [0, 1],
        "rank_in_group": 0,
        "layer_call_id": 0,
        "logical_width": 8,
        "tensor_rows": 3,
        "element_size": 2,
    }
