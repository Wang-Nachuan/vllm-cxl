# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.distributed.ccl_profiler as ccl_profiler


def test_count_only_iteration_needs_no_cuda_event_pool() -> None:
    profiler = ccl_profiler.CCLProfiler()
    profiler.start_iteration(
        phase="prefill",
        rank=0,
        batch_size=1,
        input_len=4,
        output_len=1,
        max_events=0,
    )
    assert profiler.select_forward({"request": 4})
    forward = profiler.begin_forward()
    first = profiler.begin_collective({})
    second = profiler.begin_collective({})
    profiler.end_collective(first)
    profiler.end_collective(second)
    profiler.end_forward(forward)

    result = profiler.finish_iteration()

    assert result == {"rank": 0, "events_used": 3, "records": []}


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
        {"group_ranks": [0, 1], "input_bytes": 64}, {}
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


def test_moe_all_reduce_records_exact_source_ranks_per_token() -> None:
    profiler = ccl_profiler.CCLProfiler()
    profiler._rank = 2
    profiler._layers[0] = ccl_profiler._MoELayerCall(
        call_id=0,
        ep_size=4,
        num_experts=4,
        placement="linear",
        activation_width=4,
        activation_element_size=2,
        has_shared_experts=False,
        ep_group_ranks=[0, 1, 2, 3],
        routes=torch.tensor([[2], [2], [2]], dtype=torch.int32),
        route_source_sizes=[2, 1],
        route_logical_source_sizes=[1, 1],
        route_group_ranks=[0, 2],
    )
    record = {
        "group_ranks": [2, 3],
        "rank_in_group": 0,
        "layer_call_id": 0,
        "logical_width": 4,
        "tensor_rows": 1,
        "element_size": 2,
    }

    pattern = profiler._resolve_all_reduce(record, {})

    assert pattern == (8, b"\x01", None)


def test_ep_metadata_snapshots_pattern_derivation_inputs() -> None:
    profiler = ccl_profiler.CCLProfiler()
    profiler._forward_active = True
    profiler._current_ep_direction = "dispatch"
    profiler._current_layer_call_id = 0
    profiler._layers[0] = ccl_profiler._MoELayerCall(
        call_id=0,
        ep_size=2,
        num_experts=4,
        placement="linear",
        activation_width=8,
        activation_element_size=2,
        has_shared_experts=False,
        ep_group_ranks=[0, 1],
    )
    profiler._current_ep_logical_source_sizes = [2, 1]

    metadata = profiler.ep_metadata(
        sizes=[2, 1],
        group_ranks=[0, 1],
        rank_in_group=0,
    )

    assert metadata == {
        "op": "all_to_all",
        "direction": "dispatch",
        "group_ranks": [0, 1],
        "rank_in_group": 0,
        "layer_call_id": 0,
        "source_sizes": [2, 1],
        "logical_source_sizes": [2, 1],
    }


def test_resolved_event_records_do_not_repeat_rank_or_sequence() -> None:
    class FakeEvent:
        def elapsed_time(self, end: object) -> float:
            return 1.25

    profiler = ccl_profiler.CCLProfiler()
    profiler._event_pairs = [(FakeEvent(), FakeEvent())]  # type: ignore[list-item]
    profiler._metadata_slots = [
        {
            "op": "forward",
            "forward_id": 0,
        }
    ]

    record = profiler._resolve_slot(0, {})

    assert record == {"op": "forward", "forward_id": 0, "elapsed_ms": 1.25}
