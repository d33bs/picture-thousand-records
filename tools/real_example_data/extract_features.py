"""Segment the real ExampleHuman field and extract real compartment tables.

This is a one-off data-preparation script, not part of the installed package.
It starts from the same raw channels as the real CellProfiler
`ExampleHuman.cppipe` (see SOURCE_README.md and that file, both in this
directory). It keeps the CellProfiler-style table names and measurement
columns, but uses Cellpose for the main cell-body segmentation:

    IdentifyPrimaryObjects(DNA)   -> Nuclei   (Minimum Cross-Entropy threshold,
                                                intensity-based declumping)
    IdentifyPrimaryObjects(PH3)  -> PH3       (same, smaller objects)
    RelateObjects(Nuclei, PH3)   -> PH3.Parent_Nuclei
    Cellpose(cellbody)            -> Cells
    RelateObjects(Cells, PH3)     -> PH3.Parent_Cells
    Cells minus Nuclei            -> Cytoplasm
    MeasureObjectIntensity(DNA, PH3) on Nuclei, Cells, Cytoplasm
    MeasureObjectSizeShape(+Zernike) on Nuclei, Cells, Cytoplasm

The nuclei and PH3 objects still use scikit-image approximations of the
CellProfiler identify modules. Cellpose gives the cell-body masks used for
cell features and crops. `cp_measure`, a modern pure-Python implementation
of CellProfiler measurement math, computes the same style of
`AreaShape_*`/`Intensity_*_<channel>`/`Location_*` columns and the
`ImageNumber`/`ObjectNumber`/`Parent_*` keys.

Run with:

    uv run --group real_data python3 tools/real_example_data/extract_features.py
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pillow_jxl  # noqa: F401  (registers the JXL codec with Pillow)
from cellpose import models
from cp_measure.bulk import get_core_measurements
from PIL import Image
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

HERE = Path(__file__).parent
IMAGE_DIR = HERE / "images"
OUTPUT_DIR = HERE / "output"
IMAGE_ID = "AS_09125_050116030001_D03f00"
IMAGE_NUMBER = 1

NUCLEI_MIN_AREA = 40
PH3_MIN_AREA = 15
MIN_PEAK_DISTANCE = 7  # matches ExampleHuman.cppipe's declumping distance

# The real ExampleHuman.cppipe runs MeasureObjectSizeShape with
# "Calculate the advanced features?: No", which excludes the extra
# moments/inertia-tensor features cp_measure's sizeshape reports by default.
AREASHAPE_KEYS = [
    "Area",
    "BoundingBoxArea",
    "BoundingBoxMaximum_X",
    "BoundingBoxMaximum_Y",
    "BoundingBoxMinimum_X",
    "BoundingBoxMinimum_Y",
    "Compactness",
    "Eccentricity",
    "EquivalentDiameter",
    "EulerNumber",
    "Extent",
    "FormFactor",
    "MajorAxisLength",
    "MaximumRadius",
    "MeanRadius",
    "MedianRadius",
    "MinorAxisLength",
    "Orientation",
    "Perimeter",
    "Solidity",
]
INTENSITY_CHANNELS = {"DNA": "d0", "PH3": "d1"}
CHANNEL_SUFFIXES = {"DNA": "d0", "PH3": "d1", "cellbody": "d2"}
CROP_PADDING = 4
SOURCE = "CellProfiler ExampleHuman tutorial dataset (CC-0)"
SOURCE_URL = (
    "https://github.com/cytomining/CytoTable/tree/main/tests/"
    "data/cellprofiler/ExampleHuman"
)


def _load_channel(suffix: str) -> np.ndarray:
    """Load one 8-bit channel, normalized to CellProfiler's [0, 1] float range."""

    path = IMAGE_DIR / f"{IMAGE_ID}{suffix}.tif"
    return np.array(Image.open(path).convert("L")).astype(np.float64) / 255.0


def _load_channel_uint8(suffix: str) -> np.ndarray:
    path = IMAGE_DIR / f"{IMAGE_ID}{suffix}.tif"
    return np.array(Image.open(path).convert("L"))


def _declump_by_intensity(
    intensity: np.ndarray,
    foreground: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    """Approximate CellProfiler's "Intensity" declumping method: watershed
    seeded from smoothed-intensity local maxima, splitting on the (inverted)
    smoothed intensity itself rather than a distance transform.
    """

    smoothed = gaussian(intensity, sigma=1.3488)
    peaks = peak_local_max(smoothed, min_distance=min_distance, labels=foreground)
    peak_mask = np.zeros_like(foreground, dtype=bool)
    peak_mask[tuple(peaks.T)] = True
    seeds = label(peak_mask)
    return watershed(-smoothed, seeds, mask=foreground)


def _segment_primary(
    channel: np.ndarray,
    min_area: int,
    min_distance: int,
    threshold_func: Callable[[np.ndarray], float] = threshold_li,
) -> np.ndarray:
    """Approximate IdentifyPrimaryObjects: a global threshold plus
    intensity-based declumping.

    The real pipeline's Minimum-Cross-Entropy threshold (`threshold_li`) is a
    good match for the DNA channel, but degenerates on PH3 -- a channel that
    is almost entirely background with a few sparse, bright mitotic foci --
    where Li's method picks up ~30% of pixels as foreground instead of the
    expected sub-1%. Otsu handles that minority-class case correctly, so PH3
    segmentation passes `threshold_otsu` here instead.
    """

    foreground = channel > threshold_func(channel)
    foreground = remove_small_objects(foreground, min_size=min_area)
    return _declump_by_intensity(channel, foreground, min_distance)


def _segment_cellpose_cells(cellbody: np.ndarray) -> np.ndarray:
    """Segment cell bodies with the default Cellpose model.

    Cellpose expects image-like intensity values, so this uses the raw uint8
    cell-body channel rather than the [0, 1] normalized channel used by the
    CellProfiler-style thresholding helpers.
    """

    model = models.CellposeModel(gpu=False)
    return model.eval(cellbody, channel_axis=None, diameter=None, min_size=40)[0]


def _relate_to_nearest_object(
    child_mask: np.ndarray,
    parent_mask: np.ndarray,
) -> dict[int, int]:
    """Approximate RelateObjects: each child is related to whichever parent
    it mostly overlaps, or the nearest parent centroid
    if it does not overlap one at all.
    """

    parent_centroids = {p.label: p.centroid for p in regionprops(parent_mask)}
    parents = {}
    for prop in regionprops(child_mask):
        coords = prop.coords
        overlap = parent_mask[coords[:, 0], coords[:, 1]]
        overlap = overlap[overlap > 0]
        if overlap.size:
            parents[prop.label] = int(np.bincount(overlap).argmax())
        else:
            cy, cx = prop.centroid
            parents[prop.label] = min(
                parent_centroids,
                key=lambda i: (parent_centroids[i][0] - cy) ** 2
                + (parent_centroids[i][1] - cx) ** 2,
            )
    return parents


def _object_numbers(mask: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(mask) if v != 0)


def _measure_compartment(
    mask: np.ndarray,
    channels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Run cp_measure over one compartment, matching ExampleHuman.cppipe's
    MeasureObjectSizeShape (+Zernike) and MeasureObjectIntensity(DNA, PH3).
    """

    object_numbers = _object_numbers(mask)
    measurements = get_core_measurements()
    reference_image = next(iter(channels.values()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape = measurements["sizeshape"](mask, reference_image)
        zernike = measurements["zernike"](mask, reference_image)
        feret = measurements["feret"](mask, reference_image)

    for key, values in shape.items():
        assert len(values) == len(object_numbers), key

    columns: dict[str, np.ndarray] = {
        "ImageNumber": np.full(len(object_numbers), IMAGE_NUMBER),
        "ObjectNumber": np.array(object_numbers),
    }
    for key in AREASHAPE_KEYS:
        columns[f"AreaShape_{key}"] = shape[key]
    columns["AreaShape_MaxFeretDiameter"] = feret["MaxFeretDiameter"]
    columns["AreaShape_MinFeretDiameter"] = feret["MinFeretDiameter"]
    for key, values in zernike.items():
        columns[f"AreaShape_{key}"] = values
    columns["Location_Center_X"] = shape["Center_X"]
    columns["Location_Center_Y"] = shape["Center_Y"]
    columns["Location_Center_Z"] = np.zeros(len(object_numbers))

    for channel_name, image in channels.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity = measurements["intensity"](mask, image)
        for key, values in intensity.items():
            columns[f"{key}_{channel_name}"] = values

    return pd.DataFrame(columns)


def _encode_jxl(crop: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(crop, mode="L").save(buffer, format="JXL")
    return buffer.getvalue()


def _encode_jpeg(crop: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(crop, mode="L").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _object_crops(
    cells_df: pd.DataFrame,
    channel_images: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Crop each real segmented cell out of each raw channel, approximating
    an OME-Arrow tile record scoped to a single object instead of a uniform
    image-wide tile grid. Each row is a self-describing OME-Arrow-style
    record: the source field's own dimensions (size_c/z/y/x) alongside this
    crop's own location and size (tile_y/tile_x/height/width), with the
    pixel data stored three ways -- a raw uncompressed array (queryable with
    no decoding), JPEG XL (a compact modern encoding), and plain JPEG (for
    universal compatibility, matching this project's own polyglot format).
    """

    size_y, size_x = next(iter(channel_images.values())).shape
    rows = []
    for _, cell in cells_df.iterrows():
        y0 = max(0, int(cell["AreaShape_BoundingBoxMinimum_Y"]) - CROP_PADDING)
        x0 = max(0, int(cell["AreaShape_BoundingBoxMinimum_X"]) - CROP_PADDING)
        y1 = min(size_y, int(cell["AreaShape_BoundingBoxMaximum_Y"]) + CROP_PADDING)
        x1 = min(size_x, int(cell["AreaShape_BoundingBoxMaximum_X"]) + CROP_PADDING)
        for channel_name, image in channel_images.items():
            crop = image[y0:y1, x0:x1]
            rows.append(
                {
                    "image_id": IMAGE_ID,
                    "object_type": "cells",
                    "ObjectNumber": int(cell["ObjectNumber"]),
                    "channel": channel_name,
                    "z": 0,
                    "tile_y": y0,
                    "tile_x": x0,
                    "height": crop.shape[0],
                    "width": crop.shape[1],
                    "dtype": "uint8",
                    "size_c": len(channel_images),
                    "size_z": 1,
                    "size_y": size_y,
                    "size_x": size_x,
                    "source": SOURCE,
                    "source_url": SOURCE_URL,
                    "pixel_data_raw": crop.tobytes(),
                    "pixel_data_jpegxl": _encode_jxl(crop),
                    "pixel_data_jpeg": _encode_jpeg(crop),
                }
            )
    return pd.DataFrame(rows)


def _image_metadata() -> pd.DataFrame:
    sample_channel = _load_channel_uint8(CHANNEL_SUFFIXES["DNA"])
    return pd.DataFrame(
        [
            {
                "image_id": IMAGE_ID,
                "size_c": len(CHANNEL_SUFFIXES),
                "size_y": sample_channel.shape[0],
                "size_x": sample_channel.shape[1],
                "dtype": "uint8",
                "channel_names": list(CHANNEL_SUFFIXES.keys()),
                "source": SOURCE,
                "source_url": SOURCE_URL,
            }
        ]
    )


def main() -> None:
    dna = _load_channel("d0")
    ph3 = _load_channel("d1")
    cellbody = _load_channel_uint8("d2")
    channels = {"DNA": dna, "PH3": ph3}

    nuclei = _segment_primary(dna, NUCLEI_MIN_AREA, MIN_PEAK_DISTANCE)
    ph3_objects = _segment_primary(
        ph3, PH3_MIN_AREA, min_distance=4, threshold_func=threshold_otsu
    )
    cells = _segment_cellpose_cells(cellbody)
    cytoplasm = np.where(nuclei > 0, 0, cells)

    print(
        f"segmented {nuclei.max()} nuclei, {cells.max()} cells, "
        f"{ph3_objects.max()} PH3 foci"
    )

    nuclei_df = _measure_compartment(nuclei, channels)
    cells_df = _measure_compartment(cells, channels)
    cytoplasm_df = _measure_compartment(cytoplasm, channels)
    cell_parents = _relate_to_nearest_object(cells, nuclei)
    cells_df["Parent_Nuclei"] = cells_df["ObjectNumber"].map(cell_parents)
    cytoplasm_df["Parent_Nuclei"] = cytoplasm_df["ObjectNumber"].map(cell_parents)
    cytoplasm_df["Parent_Cells"] = cytoplasm_df["ObjectNumber"]

    ph3_nuclei_parents = _relate_to_nearest_object(ph3_objects, nuclei)
    ph3_cell_parents = _relate_to_nearest_object(ph3_objects, cells)
    ph3_numbers = _object_numbers(ph3_objects)
    ph3_props = {p.label: p.centroid for p in regionprops(ph3_objects)}
    ph3_df = pd.DataFrame(
        {
            "ImageNumber": IMAGE_NUMBER,
            "ObjectNumber": ph3_numbers,
            "Location_Center_X": [ph3_props[n][1] for n in ph3_numbers],
            "Location_Center_Y": [ph3_props[n][0] for n in ph3_numbers],
            "Location_Center_Z": 0.0,
            "Parent_Nuclei": [ph3_nuclei_parents[n] for n in ph3_numbers],
            "Parent_Cells": [ph3_cell_parents[n] for n in ph3_numbers],
        }
    )

    channel_images = {
        name: _load_channel_uint8(suffix) for name, suffix in CHANNEL_SUFFIXES.items()
    }
    object_crops_df = _object_crops(cells_df, channel_images)
    image_metadata_df = _image_metadata()

    OUTPUT_DIR.mkdir(exist_ok=True)
    tables = {
        "nuclei": nuclei_df,
        "cells": cells_df,
        "cytoplasm": cytoplasm_df,
        "ph3": ph3_df,
        "images_metadata": image_metadata_df,
        "object_crops": object_crops_df,
    }
    for name, frame in tables.items():
        out_path = OUTPUT_DIR / f"real_{name}.parquet"
        frame.to_parquet(out_path, index=False)
        print(f"wrote {len(frame)} rows, {len(frame.columns)} columns -> {out_path}")


if __name__ == "__main__":
    main()
