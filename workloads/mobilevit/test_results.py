from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


RESULTS = Path(__file__).with_name("results.py")
SPEC = importlib.util.spec_from_file_location("mobilevit_results", RESULTS)
assert SPEC and SPEC.loader
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


def test_timings_writes_operation_and_transition_reports(tmp_path: Path) -> None:
    application = tmp_path / "application"
    tiles = application / "src" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / "tile_0.c").write_text(
        """.hartid = 0u,
static const op_desc_t plan_ops[] = {
  { .kind = OP_GROUP_NORMALIZE, },
};
static const fifo_send_desc_t plan_sends[] = {
  { .transition_id = 7, .dst_hartid = 1, .src = { .num_elems = 8u, .elem_bytes = 2u, }, },
};
""",
        encoding="utf-8",
    )
    (tiles / "tile_1.c").write_text(
        """.hartid = 1u,
static const fifo_recv_desc_t plan_recvs[] = {
  { .transition_id = 7, .src_hartid = 0, },
};
""",
        encoding="utf-8",
    )
    log = tmp_path / "maps.log"
    log.write_text(
        "maps t0 tok 0 slot 0 op 0 cycles 11\n"
        "maps t0 tok 0 slot 0 send 7 cycles 5\n"
        "maps t1 tok 0 slot 0 recv 7 cycles 9\n"
        "maps t0 tok 1 slot 1 op 0 cycles 13\n"
        "maps t0 tok 1 slot 1 send 7 cycles 6\n"
        "maps t1 tok 1 slot 1 recv 7 cycles 8\n",
        encoding="utf-8",
    )
    execution_plan = tmp_path / "execution-plan.json"
    execution_plan.write_text(json.dumps({"stages": [
        {"submesh": {"tile_ids": [0]}},
        {"submesh": {"tile_ids": [1]}},
    ]}), encoding="utf-8")
    output = tmp_path / "timings"
    results.timings(type("Arguments", (), {
        "application": application, "maps_log": log, "output": output,
        "execution_plan": execution_plan,
    })())

    with (output / "tile_0.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{
            "tile": "0", "token": "0", "slot": "0", "operation_index": "0",
            "operation_kind": "OP_GROUP_NORMALIZE", "duration_cycles": "11",
        }, {
            "tile": "0", "token": "1", "slot": "1", "operation_index": "0",
            "operation_kind": "OP_GROUP_NORMALIZE", "duration_cycles": "13",
        }]
    with (output / "transitions.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{
            "transition_id": "7", "source_tile": "0", "destination_tile": "1",
            "token": "0", "slot": "0", "payload_bytes": "16",
            "send_start_cycle": "", "send_end_cycle": "", "send_idma_duration_cycles": "5",
            "receiver_wait_start_cycle": "", "receiver_visible_cycle": "",
            "receiver_visibility_wait_cycles": "9",
        }, {
            "transition_id": "7", "source_tile": "0", "destination_tile": "1",
            "token": "1", "slot": "1", "payload_bytes": "16",
            "send_start_cycle": "", "send_end_cycle": "", "send_idma_duration_cycles": "6",
            "receiver_wait_start_cycle": "", "receiver_visible_cycle": "",
            "receiver_visibility_wait_cycles": "8",
        }]
    with (output / "stage-maxima.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{
            "row_kind": "kernel", "token": "0", "stage_id": "0",
            "source_stage_id": "", "destination_stage_id": "", "operation_index": "0",
            "operation_kind": "OP_GROUP_NORMALIZE", "transition_id": "",
            "max_duration_cycles": "11", "max_send_idma_duration_cycles": "",
            "max_receiver_visibility_wait_cycles": "",
        }, {
            "row_kind": "kernel", "token": "1", "stage_id": "0",
            "source_stage_id": "", "destination_stage_id": "", "operation_index": "0",
            "operation_kind": "OP_GROUP_NORMALIZE", "transition_id": "",
            "max_duration_cycles": "13", "max_send_idma_duration_cycles": "",
            "max_receiver_visibility_wait_cycles": "",
        }, {
            "row_kind": "transition", "token": "0", "stage_id": "",
            "source_stage_id": "0", "destination_stage_id": "1", "operation_index": "",
            "operation_kind": "", "transition_id": "7", "max_duration_cycles": "",
            "max_send_idma_duration_cycles": "5", "max_receiver_visibility_wait_cycles": "9",
        }, {
            "row_kind": "transition", "token": "1", "stage_id": "",
            "source_stage_id": "0", "destination_stage_id": "1", "operation_index": "",
            "operation_kind": "", "transition_id": "7", "max_duration_cycles": "",
            "max_send_idma_duration_cycles": "6", "max_receiver_visibility_wait_cycles": "8",
        }]


def test_timings_ignores_legacy_nonreceive_events(tmp_path: Path) -> None:
    application = tmp_path / "application"
    tiles = application / "src" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / "tile_0.c").write_text(
        ".hartid = 0u,\n"
        "static const fifo_send_desc_t plan_sends[] = {\n"
        "  { .transition_id = 2, .dst_hartid = 1, .src = { .num_elems = 8u, .elem_bytes = 2u, }, },\n"
        "};\n",
        encoding="utf-8",
    )
    (tiles / "tile_1.c").write_text(
        ".hartid = 1u,\n"
        "static const fifo_recv_desc_t plan_recvs[] = {\n"
        "  { .transition_id = 2, .src_hartid = 0, },\n"
        "};\n",
        encoding="utf-8",
    )
    log = tmp_path / "maps.log"
    log.write_text(
        "MAPS_TOKEN tile=0 token=0 start=10 end=50 output=0\n"
        "MAPS_TOKEN tile=1 token=0 start=10 end=50 output=0\n"
        "MAPS_TOKEN tile=0 token=1 start=60 end=100 output=0\n"
        "MAPS_TOKEN tile=1 token=1 start=60 end=100 output=0\n"
        "maps t1 tok 0 slot 0 recv 0 start 11 end 15 cycles 4\n"
        "maps t0 tok 0 slot 0 send 2 start 20 end 25 cycles 5\n"
        "maps t1 tok 0 slot 0 recv 2 start 18 end 30 cycles 12\n"
        "maps t1 tok 0 slot 0 recv 2 start 9 end 105 cycles 96\n"
        "maps t0 tok 1 slot 1 send 2 start 70 end 75 cycles 5\n"
        "maps t1 tok 1 slot 1 recv 2 start 68 end 80 cycles 12\n"
        "maps t1 tok 1 slot 1 l2-read 0 start 61 end 64 cycles 3\n",
        encoding="utf-8",
    )
    execution_plan = tmp_path / "execution-plan.json"
    execution_plan.write_text(json.dumps({"stages": [
        {"submesh": {"tile_ids": [0]}},
        {"submesh": {"tile_ids": [1]}},
    ]}), encoding="utf-8")
    output = tmp_path / "timings"

    results.timings(type("Arguments", (), {
        "application": application, "maps_log": log, "output": output,
        "execution_plan": execution_plan,
    })())

    with (output / "transitions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["transition_id"] for row in rows} == {"2"}
    assert {row["receiver_visibility_wait_cycles"] for row in rows} == {"12"}
    assert {row["payload_bytes"] for row in rows} == {"16"}


def test_collect_keeps_all_four_execution_tokens(tmp_path: Path) -> None:
    maps_log = tmp_path / "maps.log"
    maps_log.write_text("\n".join(
        f"MAPS_TOKEN tile={tile} token={token} start={20 * token + 5 * tile} "
        f"end={20 * (token + 1) + 5 * tile} output={int(tile == 0)}"
        for token in range(4)
        for tile in range(2)
    ) + "\n", encoding="utf-8")
    full_mesh_log = tmp_path / "full-mesh.log"
    full_mesh_log.write_text(
        "REFERENCE_TILES count=16\n" + "".join(
            f"REFERENCE_COMPLETION token={token} cycle={100 * (token + 1)}\n"
            for token in range(4)
        ), encoding="utf-8",
    )
    checksums = tmp_path / "checksums.csv"
    checksums.write_text("", encoding="utf-8")
    references_available = tmp_path / "references-available.txt"
    references_available.write_text("0", encoding="utf-8")
    output = tmp_path / "results.csv"

    results.collect(type("Arguments", (), {
        "maps_log": maps_log, "full_mesh_log": full_mesh_log,
        "reference_checksums": checksums,
        "reference_available": references_available,
        "mesh": "4x4", "tokens": 4, "csv": output,
    })())

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [int(row["token"]) for row in rows if row["strategy"] == "maps"] == [0, 1, 2, 3]
    assert [int(row["token"]) for row in rows if row["strategy"] == "full-mesh"] == [0, 1, 2, 3]


def test_timings_writes_worst_tile_stage_totals(tmp_path: Path) -> None:
    application = tmp_path / "application"
    tiles = application / "src" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / "tile_0.c").write_text(
        """.hartid = 0u,
static const op_desc_t plan_ops[] = {
  { .kind = OP_ADD, },
  { .kind = OP_MUL, },
};
static const fifo_send_desc_t plan_sends[] = {
  { .transition_id = 7, .dst_hartid = 2, .src = { .num_elems = 8u, .elem_bytes = 2u, }, },
  { .transition_id = 8, .dst_hartid = 2, .src = { .num_elems = 8u, .elem_bytes = 2u, }, },
};
""",
        encoding="utf-8",
    )
    (tiles / "tile_1.c").write_text(
        """.hartid = 1u,
static const op_desc_t plan_ops[] = {
  { .kind = OP_RELU, },
};
static const fifo_send_desc_t plan_sends[] = {
  { .transition_id = 9, .dst_hartid = 2, .src = { .num_elems = 8u, .elem_bytes = 2u, }, },
};
""",
        encoding="utf-8",
    )
    (tiles / "tile_2.c").write_text(
        """.hartid = 2u,
static const fifo_recv_desc_t plan_recvs[] = {
  { .transition_id = 7, .src_hartid = 0, },
  { .transition_id = 8, .src_hartid = 0, },
  { .transition_id = 9, .src_hartid = 1, },
};
""",
        encoding="utf-8",
    )
    log = tmp_path / "maps.log"
    log.write_text(
        "maps t0 tok 0 slot 0 op 0 cycles 11\n"
        "maps t0 tok 0 slot 0 op 1 cycles 7\n"
        "maps t0 tok 0 slot 0 send 7 cycles 5\n"
        "maps t0 tok 0 slot 0 send 8 cycles 7\n"
        "maps t1 tok 0 slot 0 op 0 cycles 20\n"
        "maps t1 tok 0 slot 0 send 9 cycles 10\n"
        "maps t2 tok 0 slot 0 recv 7 cycles 3\n"
        "maps t2 tok 0 slot 0 recv 8 cycles 4\n"
        "maps t2 tok 0 slot 0 recv 9 cycles 6\n",
        encoding="utf-8",
    )
    execution_plan = tmp_path / "execution-plan.json"
    execution_plan.write_text(json.dumps({"stages": [
        {"submesh": {"tile_ids": [0, 1]}},
        {"submesh": {"tile_ids": [2]}},
    ]}), encoding="utf-8")
    output = tmp_path / "timings"

    results.timings(type("Arguments", (), {
        "application": application, "maps_log": log, "output": output,
        "execution_plan": execution_plan,
    })())

    with (output / "stage-totals.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{
            "token": "0", "stage_id": "0",
            "worst_compute_tile": "1", "compute_cycles": "20",
            "operation_count": "1", "worst_transition_tile": "0",
            "transition_cycles": "12", "write_count": "2",
        }, {
            "token": "0", "stage_id": "1",
            "worst_compute_tile": "2", "compute_cycles": "0",
            "operation_count": "0", "worst_transition_tile": "2",
            "transition_cycles": "0", "write_count": "0",
        }]
