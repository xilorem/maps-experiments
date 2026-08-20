"""Deterministic MobileViT nodes 170 through 179 workload artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference, utils


FIRST_NODE_INDEX = 170
LAST_NODE_INDEX = 179
INPUT_SHAPE = (1, 128, 4, 16)
INPUT_SEED = 170179
TOKEN_COUNT = 2
TOKEN_SLOTS = 2
REFERENCE_ATOL = 0.5
REFERENCE_RTOL = 0.05


def _specialize_fp16(model: onnx.ModelProto) -> None:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT:
            tensor_type.elem_type = TensorProto.FLOAT16
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            converted = numpy_helper.from_array(
                numpy_helper.to_array(initializer).astype(np.float16),
                name=initializer.name,
            )
            initializer.CopyFrom(converted)


def extract_mobilevit_slice(source: Path, output: Path) -> Path:
    """Extract the agreed graph slice with its original Initializers."""

    model = shape_inference.infer_shapes(onnx.load(source))
    if len(model.graph.node) <= LAST_NODE_INDEX:
        raise ValueError("source model does not contain MobileViT nodes 170 through 179")
    output.parent.mkdir(parents=True, exist_ok=True)
    utils.extract_model(
        source,
        output,
        [model.graph.node[FIRST_NODE_INDEX].input[0]],
        [model.graph.node[LAST_NODE_INDEX].output[0]],
        check_model=True,
        infer_shapes=True,
    )
    extracted = onnx.load(output)
    extracted.graph.name = "mobilevit_nodes_170_179"
    _specialize_fp16(extracted)
    onnx.checker.check_model(extracted)
    onnx.save(extracted, output)
    return output


def generate_runtime_inputs(output: Path, token_count: int = TOKEN_COUNT) -> Path:
    """Write complete, ordered, distinct deterministic FP16 token values."""

    values = np.random.default_rng(INPUT_SEED).uniform(
        -1.0,
        1.0,
        size=(token_count, *INPUT_SHAPE),
    ).astype(np.dtype("<f2"))
    if len({token.tobytes() for token in values}) != token_count:
        raise RuntimeError("deterministic Runtime Inputs are not distinct")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(values.tobytes())
    return output


def generate_onnx_references(
    model_path: Path,
    input_path: Path,
    output: Path,
    token_count: int = TOKEN_COUNT,
) -> Path:
    """Evaluate one ONNX Runtime reference output per Execution Token."""

    import onnxruntime as ort

    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    values = np.fromfile(input_path, dtype=np.dtype("<f2")).reshape(
        (token_count, *INPUT_SHAPE)
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    references = np.stack(
        [session.run(None, {input_name: token})[0] for token in values]
    ).astype(np.dtype("<f2"))
    if references.shape != (token_count, *INPUT_SHAPE):
        raise RuntimeError(f"unexpected ONNX Runtime output shape: {references.shape}")
    if not np.isfinite(references).all():
        raise RuntimeError("ONNX Runtime produced non-finite reference output")
    output.write_bytes(references.tobytes())
    return output
