"""Test fixtures.

API tests must not depend on multi-megabyte trained weights, so we build a tiny ONNX
"fake detector" with the exact same input/output signature as a YOLO11 export:
  input  "images" float32 (1, 3, 640, 640)
  output "output0" float32 (1, 4 + C, N)   - here a constant tensor with 2 known boxes
and the Ultralytics-style "names" metadata. This tests the whole serving path
(upload -> decode -> letterbox -> ONNX Runtime -> NMS -> rescale -> DB) deterministically.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

NAMES = {0: "person", 1: "car"}


def make_fake_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    C, N = len(NAMES), 8
    out = np.zeros((1, 4 + C, N), np.float32)
    out[0, :4, 0] = [160, 320, 64, 128]; out[0, 4 + 0, 0] = 0.92    # a person (network coords)
    out[0, :4, 1] = [162, 321, 64, 128]; out[0, 4 + 0, 1] = 0.60    # duplicate -> removed by NMS
    out[0, :4, 2] = [480, 320, 160, 80]; out[0, 4 + 1, 2] = 0.81    # a car
    const = numpy_helper.from_array(out, "boxes_const")
    # output = const + 0 * mean(images): keeps the input connected to the graph
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["boxes_const", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + C, N])],
        initializer=[const, numpy_helper.from_array(np.array(0, np.float32), "zero")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    helper.set_model_props(model, {"names": str(NAMES)})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("va")
    (tmp / "models").mkdir()
    make_fake_model(tmp / "models" / "fake.onnx")
    os.environ.update({
        "VA_DATABASE_URL": f"sqlite:///{(tmp / 'test.db').as_posix()}",
        "VA_MODELS_DIR": str(tmp / "models"),
        "VA_DATA_DIR": str(tmp / "data"),
        "VA_METRICS_FILE": str(tmp / "metrics.json"),
        "VA_DEFAULT_MODEL": "fake",
        "VA_DEVICE": "cpu",
    })
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
