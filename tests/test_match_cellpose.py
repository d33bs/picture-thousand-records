"""Unit tests for the Cellpose-to-manual matching in match_cellpose.py."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tools.bbbc039_data.match_cellpose import _match_objects


def _labels(
    shapes: list[tuple[int, int, int, int]],
    shape: tuple[int, int] = (30, 30),
) -> np.ndarray:
    """Build a label image with one filled rectangle per (y0, x0, y1, x1)."""

    labels = np.zeros(shape, dtype=np.int32)
    for label_id, (y0, x0, y1, x1) in enumerate(shapes, start=1):
        labels[y0:y1, x0:x1] = label_id
    return labels


def test_perfect_overlap_is_captured_with_iou_one() -> None:
    manual = _labels([(2, 2, 12, 12)])
    cellpose = _labels([(2, 2, 12, 12)])
    captured, iou = _match_objects(manual, cellpose)
    assert captured[1]
    assert iou[1] == pytest.approx(1.0)


def test_iou_exactly_at_threshold_is_captured() -> None:
    # Manual object is 10x10; the Cellpose mask adds a 10x10 ring, so
    # intersection = 100 and union = 200 -> IoU exactly 0.5.
    manual = _labels([(5, 5, 15, 15)])
    cellpose = _labels([(5, 5, 15, 25)])
    captured, iou = _match_objects(manual, cellpose)
    assert captured[1]
    assert iou[1] == pytest.approx(0.5)


def test_iou_just_below_threshold_is_not_captured() -> None:
    manual = _labels([(5, 5, 15, 15)])
    cellpose = _labels([(5, 5, 15, 26)])  # 100 / 211 < 0.5
    captured, iou = _match_objects(manual, cellpose)
    assert not captured[1]
    assert 0 < iou[1] < 0.5


def test_two_manual_objects_competing_for_one_mask() -> None:
    # Both manual objects overlap the single Cellpose mask equally; only the
    # pair with the higher IoU may capture it, so one object stays missed.
    manual = _labels([(2, 2, 10, 10), (12, 12, 20, 20)])
    cellpose = _labels([(2, 2, 11, 11)])  # overlaps both, better with the first
    captured, iou = _match_objects(manual, cellpose)
    assert captured[1]
    assert captured.sum() == 1
    assert not captured[2]
    assert iou[1] >= iou[2]


def test_one_cellpose_mask_never_matches_twice() -> None:
    manual = _labels([(0, 0, 12, 12), (12, 0, 18, 12)])
    cellpose = _labels([(0, 0, 24, 12)])  # one mask covering both objects
    captured, iou = _match_objects(manual, cellpose)
    assert captured.sum() == 1
    # The reported IoU of a missed object is its best overlap with any mask.
    assert 0 < iou[2] < 0.5


def test_no_overlap_reports_zero_iou() -> None:
    manual = _labels([(0, 0, 10, 10)])
    cellpose = _labels([(20, 20, 30, 30)])
    captured, iou = _match_objects(manual, cellpose)
    assert not captured[1]
    assert iou[1] == 0.0


def test_empty_either_side_captures_nothing() -> None:
    manual = _labels([(0, 0, 10, 10)])
    empty = np.zeros((30, 30), dtype=np.int32)
    captured, _ = _match_objects(manual, empty)
    assert not captured[1]
    captured, _ = _match_objects(empty, manual)
    assert captured.shape == (1,) and not captured[0]


def test_module_stays_importable_without_heavy_dependencies() -> None:
    # The heavy imports (cellpose, PIL, pandas, extract_features/cp_measure)
    # must stay inside functions so this module imports without the cellpose
    # or real_data groups installed.
    source = (
        Path(__file__).parent.parent
        / "tools"
        / "bbbc039_data"
        / "match_cellpose.py"
    ).read_text()
    heavy = {"cellpose", "PIL", "pandas", "extract_features"}
    top_level_imports = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import) and node.col_offset == 0
        for alias in node.names
        if alias.name.split(".")[0] in heavy
    } | {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0 and node.module
        if node.module.split(".")[0] in heavy
    }
    assert top_level_imports == set()
