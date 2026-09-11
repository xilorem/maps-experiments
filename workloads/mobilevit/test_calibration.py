from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


CALIBRATION = Path(__file__).with_name("calibration.py")
SPEC = importlib.util.spec_from_file_location("mobilevit_calibration", CALIBRATION)
assert SPEC and SPEC.loader
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    application = tmp_path / "application"
    tiles = application / "src" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / "tile_0.c").write_text(
        """.hartid = 0u,
static const op_desc_t plan_ops[] = {
  {
    .kind = OP_RELU,
    .num_inputs = 1u,
    .inputs = {{.shape = {8192, 0, 0, 0, 0, 0}}},
    .num_outputs = 1u,
    .outputs = {{.shape = {8192, 0, 0, 0, 0, 0}}},
    .params = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
  },
};
""",
        encoding="utf-8",
    )
    log = tmp_path / "maps.log"
    log.write_text(
        "maps t0 tok 0 slot 0 op 0 cycles 4200\n"
        "maps t0 tok 1 slot 1 op 0 start 20 end 3835 cycles 3815\n",
        encoding="utf-8",
    )
    return application, log


def test_calibration_uses_warm_measurements_and_predicts_cycles(tmp_path: Path) -> None:
    application, log = _fixture(tmp_path)

    rows = calibration.calibration_rows("4x4", application, log)

    assert len(rows) == 1
    assert rows[0]["token"] == 1
    assert rows[0]["operation_kind"] == "OP_RELU"
    assert rows[0]["operation_count"] == 8192
    assert rows[0]["predicted_cycles"] == 3807


def test_calibration_accepts_eight_by_eight_measurements(tmp_path: Path) -> None:
    application, log = _fixture(tmp_path)

    rows = calibration.calibration_rows("8x8", application, log)

    assert rows[0]["mesh"] == "8x8"


def test_calibration_subtracts_spatz_instruction_fetch_stalls(tmp_path: Path) -> None:
    application, log = _fixture(tmp_path)
    vcd = tmp_path / "imiss.vcd"
    vcd.write_text(
        """$timescale 1ps $end
$scope module magia-v3-soc $end
$scope module tile-clock $end
$var wire 64 p period $end
$upscope $end
$scope module magia-tile-0 $end
$scope module tile-0-snitch-spatz $end
$var wire 1 i event_imiss $end
$upscope $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
b101 p
0i
#125
1i
#175
0i
""",
        encoding="utf-8",
    )

    rows = calibration.calibration_rows("8x8", application, log, vcd)

    assert rows[0]["raw_measured_cycles"] == 3815
    assert rows[0]["instruction_fetch_cycles"] == 10
    assert rows[0]["measured_cycles"] == 3805
    assert rows[0]["dominance"] == "compute"


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        (
            calibration.Operation(
                "OP_MATMUL",
                ((64, 128), (128, 32)),
                ((64, 32),),
                (64, 128, 32, 1),
                0,
            ),
            86_201,
        ),
        (
            calibration.Operation(
                "OP_MUL",
                ((1, 128, 4, 16), (1, 1, 4, 16)),
                ((1, 128, 4, 16),),
                (),
                0,
            ),
            7_013,
        ),
        (
            calibration.Operation(
                "OP_REDUCE_SUM",
                ((1, 1, 4, 16),),
                ((1, 1, 4, 1),),
                (3,),
                0,
            ),
            2_668,
        ),
    ),
)
def test_predictor_uses_shape_aware_compute_formulas(
    operation: calibration.Operation,
    expected: int,
) -> None:
    assert calibration.predicted_cycles(operation) == expected


@pytest.mark.parametrize(
    ("kind", "row_len", "expected"),
    (
        ("OP_SUB", 2, 2_896),
        ("OP_SUB", 3, 4_544),
        ("OP_DIV", 2, 3_208),
        ("OP_DIV", 3, 4_901),
    ),
)
def test_predictor_models_odd_binary_row_scalar_fallback(
    kind: str, row_len: int, expected: int
) -> None:
    operation = calibration.Operation(
        kind,
        ((1, 1, 1, row_len), (1, 1, 1, 1)),
        ((1, 1, 1, row_len),),
        (),
        0,
    )

    assert calibration.predicted_cycles(operation) == expected
