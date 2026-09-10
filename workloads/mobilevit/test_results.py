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
