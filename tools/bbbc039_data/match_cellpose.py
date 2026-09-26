"""Match Cellpose detections against the BBBC039 manual nucleus annotations.

This is a data-preparation script. It runs Cellpose on the same raw images the
manual masks annotate, then records for every manually annotated nucleus
whether a Cellpose mask captured it (IoU >= 0.5, one-to-one greedy matching:
each Cellpose mask is matched to at most one manual object, best IoU wins).
The Cellpose masks themselves are not stored; only per-object
`cellpose_captured` / `cellpose_iou` columns are, in
`bbbc039_cellpose_matches.parquet`, which the builder LEFT JOINs into
`bbbc039.objects`.

The first run downloads about 1 GB of model weights and segments 200 fields.
On Apple Silicon it runs on the MPS GPU (~5 s per field, a few minutes total);
on CUDA or CPU-only machines it falls back accordingly (expect hours on CPU).
Finished masks are cached under `output/cellpose_masks/`, so interrupted runs
resume where they stopped.

Run with:

    uv run --group real_data --group cellpose \
        python tools/bbbc039_data/match_cellpose.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
MASK_CACHE_DIR = OUTPUT_DIR / "cellpose_masks"
PACKAGE_ASSETS_DIR = HERE.parent.parent / "src" / "picture_thousand_records" / "assets"

IOU_THRESHOLD = 0.5
CELLPOSE_MODEL = "cpsam_v2"
CELLPOSE_MIN_SIZE = 40
# MPS (Apple GPU) is ~20x faster than CPU for cpsam_v2 and gives the same
# masks; torch.cuda or CPU machines fall back automatically.
CELLPOSE_GPU = True
CELLPOSE_BFLOAT16 = False  # bfloat16 is not supported on MPS
MATCHES_TABLE = "bbbc039_cellpose_matches"


def _match_objects(
    manual_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    threshold: float = IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Match manual objects to Cellpose masks by IoU, one-to-one.

    A manual object is "captured" when a Cellpose mask overlaps it with IoU >=
    threshold. Each Cellpose mask captures at most one manual object (the
    pair with the highest IoU wins, processed greedily in IoU order).

    Returns `(captured, iou)` indexed by manual label (index 0 is background):
    `iou` holds the matched pair's IoU for captured objects, and otherwise the
    best IoU the object reached with any Cellpose mask (0.0 when none).
    """

    manual_count = int(manual_labels.max())
    cellpose_count = int(cellpose_labels.max())
    captured = np.zeros(manual_count + 1, dtype=bool)
    matched_iou = np.zeros(manual_count + 1, dtype=np.float32)
    best_iou = np.zeros(manual_count + 1, dtype=np.float32)
    if manual_count == 0 or cellpose_count == 0:
        return captured, best_iou

    pair = manual_labels.astype(np.int64) * (cellpose_count + 1) + cellpose_labels
    both = (manual_labels > 0) & (cellpose_labels > 0)
    overlap = np.bincount(
        pair[both], minlength=(manual_count + 1) * (cellpose_count + 1)
    ).reshape(manual_count + 1, cellpose_count + 1)
    manual_area = np.bincount(manual_labels.ravel(), minlength=manual_count + 1)
    cellpose_area = np.bincount(cellpose_labels.ravel(), minlength=cellpose_count + 1)

    inter = overlap[1:, 1:]
    union = manual_area[1:, None] + cellpose_area[None, 1:] - inter
    iou = inter / union
    best_iou[1:] = iou.max(axis=1)

    rows, cols = np.nonzero(iou >= threshold)
    order = np.argsort(-iou[rows, cols])
    manual_matched = np.zeros(manual_count + 1, dtype=bool)
    cellpose_matched = np.zeros(cellpose_count + 1, dtype=bool)
    for row, col in zip(rows[order], cols[order]):
        manual_id, cellpose_id = int(row) + 1, int(col) + 1
        if manual_matched[manual_id] or cellpose_matched[cellpose_id]:
            continue
        manual_matched[manual_id] = True
        cellpose_matched[cellpose_id] = True
        captured[manual_id] = True
        matched_iou[manual_id] = iou[row, col]
    return captured, np.where(captured, matched_iou, best_iou)


def _segment_image(image_u16: np.ndarray, image_id: str) -> np.ndarray:
    """Segment one field with Cellpose, caching the label image per image."""

    cache_path = MASK_CACHE_DIR / f"{image_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    # Lazy on purpose (noqa: PLC0415): cellpose pulls in the torch stack.
    from cellpose import models  # noqa: PLC0415

    model = models.CellposeModel(
        gpu=CELLPOSE_GPU,
        pretrained_model=CELLPOSE_MODEL,
        use_bfloat16=CELLPOSE_BFLOAT16,
    )
    labels, _, _ = model.eval(
        image_u16,
        channel_axis=None,
        diameter=None,
        min_size=CELLPOSE_MIN_SIZE,
    )
    labels = np.asarray(labels, dtype=np.int32)
    MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, labels)
    return labels


def main() -> None:
    # Lazy on purpose (noqa: PLC0415): extract_features pulls in cp_measure,
    # and pandas/pyarrow only matter for the parquet output, so
    # _match_objects stays cheap to import from tests.
    import pandas as pd  # noqa: PLC0415
    from extract_features import (  # noqa: PLC0415
        _decode_mask,
        _image_paths,
        _mask_path,
    )
    from PIL import Image  # noqa: PLC0415

    objects = pd.read_parquet(OUTPUT_DIR / "bbbc039_objects.parquet")
    keys = ["ObjectNumber", "ImageNumber", "LocalObjectNumber", "image_id"]

    rows = []
    for image_path in _image_paths():
        image_id = image_path.stem
        labels = _decode_mask(_mask_path(image_id))
        image_rows = objects[objects["image_id"] == image_id]
        if len(image_rows) != int(labels.max()):
            raise AssertionError(
                f"{image_id}: {len(image_rows)} object rows but the manual mask"
                f" has {labels.max()} labels; the LocalObjectNumber convention"
                " (cp_measure row order == connected-component label) has drifted."
            )
        image_u16 = np.array(Image.open(image_path), dtype=np.uint16)
        cellpose_labels = _segment_image(image_u16, image_id)
        captured, iou = _match_objects(labels, cellpose_labels)

        for row in image_rows.itertuples(index=False):
            rows.append(
                {
                    "ObjectNumber": int(row.ObjectNumber),
                    "ImageNumber": int(row.ImageNumber),
                    "LocalObjectNumber": int(row.LocalObjectNumber),
                    "image_id": image_id,
                    "cellpose_captured": bool(captured[row.LocalObjectNumber]),
                    "cellpose_iou": np.float32(iou[row.LocalObjectNumber]),
                }
            )
        print(
            f"{image_id}: {captured[1:].sum()}/{len(image_rows)} captured"
            f" ({len(image_rows) - captured[1:].sum()} missed)"
        )

    matches = pd.DataFrame(rows, columns=[*keys, "cellpose_captured", "cellpose_iou"])
    for directory in (OUTPUT_DIR, PACKAGE_ASSETS_DIR):
        matches.to_parquet(directory / f"{MATCHES_TABLE}.parquet", index=False)
    captured_total = int(matches["cellpose_captured"].sum())
    print(
        f"wrote {len(matches)} rows -> {MATCHES_TABLE}.parquet;"
        f" {captured_total} captured, {len(matches) - captured_total} missed"
        f" ({100 * captured_total / len(matches):.1f}%)"
    )


if __name__ == "__main__":
    main()
