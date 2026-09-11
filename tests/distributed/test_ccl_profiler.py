# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

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
        {},
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
    forward = ccl_profiler._DrainedForward(
        forward_id=0,
        layout=layout,
        elapsed_ms=1.0,
        collective_times=[],
        num_tokens_unpadded=1,
        logical_source_sizes=[1, 1],
        host_routes={},
        host_start_ns=1_000_000_000,
    )
    record = {
        "group_ranks": [2, 3],
        "rank_in_group": 0,
        "layer_call_id": 0,
        "logical_width": 4,
        "tensor_rows": 1,
        "element_size": 2,
    }
    routes = {0: torch.tensor([[2], [2], [2]], dtype=torch.int32)}

    pattern = profiler._resolve_all_reduce(record, routes, forward)

    assert pattern == (8, b"\x01", None)


def test_collective_records_are_offsets_from_forward_start() -> None:
    metadata = {
        "op": "all_reduce",
        "group_ranks": [0, 1],
        "rank_in_group": 0,
        "input_bytes": 64,
    }
    layout = ccl_profiler._GraphLayout(
        "FULL",
        "decode",
        collectives=[ccl_profiler._CollectiveLayout(metadata, None, None)],
        captured=True,
    )
    forward = ccl_profiler._DrainedForward(
        forward_id=3,
        layout=layout,
        elapsed_ms=1.5,
        collective_times=[(0.75, 1.25)],
        num_tokens_unpadded=1,
        logical_source_sizes=[],
        host_routes={},
        host_start_ns=1_234_567_890,
    )
    profiler = ccl_profiler.CCLProfiler()

    records = profiler._resolve_forward(forward)

    assert records == [
        {
            "op": "forward",
            "forward_id": 3,
            "elapsed_ms": 1.5,
            "host_start_ns": 1_234_567_890,
        },
        {
            "op": "all_reduce",
            "forward_id": 3,
            "group_ranks": [0, 1],
            "start_offset_ms": 0.75,
            "end_offset_ms": 1.25,
            "bytes_per_row": 64,
            "physical_source_rows": [1, 1],
            "source_masks": "Aw==",
        },
    ]


def test_layout_capture_reuses_external_event_nodes(monkeypatch) -> None:
    events = []

    class FakeEvent:
        def __init__(
            self, *, enable_timing: bool, external: bool = False
        ) -> None:
            assert enable_timing
            self.external = external
            self.records = 0
            events.append(self)

        def record(self) -> None:
            self.records += 1

    class FakeTensor:
        shape = (2, 4)

        @staticmethod
        def numel() -> int:
            return 8

        @staticmethod
        def element_size() -> int:
            return 2

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    profiler = ccl_profiler.CCLProfiler()
    tensor = FakeTensor()

    with profiler.prepare_layout("FULL", "decode"):
        for _ in range(2):
            metadata = profiler.all_reduce_metadata(
                tensor, group_ranks=[0, 1], rank_in_group=0
            )
            token = profiler.begin_collective(metadata)
            profiler.end_collective(token)

    with profiler.capture_layout("FULL", "decode"):
        for _ in range(2):
            metadata = profiler.all_reduce_metadata(
                tensor, group_ranks=[0, 1], rank_in_group=0
            )
            token = profiler.begin_collective(metadata)
            profiler.end_collective(token)

    collective_events = [event for event in events if event.external]
    assert len(collective_events) == 4
    assert [event.records for event in collective_events] == [2, 2, 2, 2]
    assert profiler._layouts[("FULL", "decode")].captured


def test_each_replay_is_drained_before_events_are_reused() -> None:
    class FakeEvent:
        def __init__(self, timestamp_ms: float = 0.0) -> None:
            self.timestamp_ms = timestamp_ms

        def record(self) -> None:
            pass

        def elapsed_time(self, end: "FakeEvent") -> float:
            return end.timestamp_ms - self.timestamp_ms

    collective_start = FakeEvent()
    collective_end = FakeEvent()
    metadata = {
        "op": "all_reduce",
        "group_ranks": [0, 1],
        "rank_in_group": 1,
        "input_bytes": 64,
    }
    descriptor = _BatchDescriptor(num_tokens=1, uniform=True)
    layout = ccl_profiler._GraphLayout(
        _Mode.FULL,
        descriptor,
        collectives=[
            ccl_profiler._CollectiveLayout(
                metadata, collective_start, collective_end
            )
        ],
        captured=True,
    )
    forward_start = FakeEvent()
    forward_end = FakeEvent()
    profiler = ccl_profiler.CCLProfiler()
    profiler._layouts[(_Mode.FULL, descriptor)] = layout
    profiler._forward_events = (forward_start, forward_end)  # type: ignore[assignment]
    profiler.start_iteration(
        phase="decode",
        rank=1,
        batch_size=1,
        input_len=4,
        output_len=3,
    )
    assert not profiler.select_forward({"request": 4})

    for replay, base_ms in enumerate((10.0, 30.0)):
        assert profiler.select_forward({"request": 1})
        forward_start.timestamp_ms = base_ms
        collective_start.timestamp_ms = base_ms + replay + 0.5
        collective_end.timestamp_ms = base_ms + replay + 1.0
        forward_end.timestamp_ms = base_ms + replay + 2.0
        token = profiler.begin_forward(
            cudagraph_mode=_Mode.FULL,
            batch_descriptor=descriptor,
            num_tokens_unpadded=1,
        )
        profiler.end_forward(token)
        profiler.enqueue_route_snapshot(token)
        profiler.drain_forward(token)

    result = profiler.finish_iteration()

    forward_records = [
        record for record in result["records"] if record["op"] == "forward"
    ]
    collective_records = [
        record for record in result["records"] if record["op"] == "all_reduce"
    ]
    assert [record["elapsed_ms"] for record in forward_records] == [2.0, 3.0]
    assert [record["start_offset_ms"] for record in collective_records] == [
        0.5,
        1.5,
    ]


def test_decode_preallocation_accepts_piecewise_only_graphs() -> None:
    descriptor = _BatchDescriptor(num_tokens=4, uniform=False)
    layout = ccl_profiler._GraphLayout(
        _Mode.PIECEWISE,
        descriptor,
        captured=True,
    )
    profiler = ccl_profiler.CCLProfiler()
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
    profiler = ccl_profiler.CCLProfiler()
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
    profiler = ccl_profiler.CCLProfiler()
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
