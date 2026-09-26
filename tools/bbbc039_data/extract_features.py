"""Extract BBBC039 source images, manual nuclei, measurements, and crops.

This is a data-preparation script. It starts from the public BBBC039 ZIP
files and writes parquet assets that the package can load into DuckDB.

The package assets keep only one JPEG crop per nucleus. The full crops (16-bit
pixels and masks), which `extract_feature_spaces.py` needs to compute MorphEm
embeddings, go to `output/` only. The original TIFF images and mask PNGs are
never copied into the package or the database.

Measurements come from cp_measure (`get_core_measurements(legacy=True)`),
run on the BBBC039 manual masks. cp_measure does not segment. Each image is
scaled to 0-1 by its own maximum before measuring (see `_cp_measure_frame`).

Run with:

    uv run --group real_data python tools/bbbc039_data/extract_features.py
"""

from __future__ import annotations

import csv
import io
import re
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from cp_measure.bulk import get_core_measurements
from PIL import Image
from skimage.measure import label, regionprops

HERE = Path(__file__).parent
RAW_DIR = HERE / "raw"
IMAGE_DIR = RAW_DIR / "images" / "images"
MASK_DIR = RAW_DIR / "masks" / "masks"
METADATA_DIR = RAW_DIR / "metadata" / "metadata"
OUTPUT_DIR = HERE / "output"
PACKAGE_ASSETS_DIR = HERE.parent.parent / "src" / "picture_thousand_records" / "assets"

SOURCE = "Broad Bioimage Benchmark Collection BBBC039"
SOURCE_URL = "https://bbbc.broadinstitute.org/BBBC039"
CHANNEL_NAME = "DNA"
CROP_PADDING = 6
REPRESENTATIVE_IMAGE_ID = "IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB"
CP_MEASUREMENTS = get_core_measurements(legacy=True)

IMAGE_RE = re.compile(
    r"^(?P<experiment>IXMtest)_(?P<well>[A-P][0-9]{2})_s(?P<site>[0-9])_w1"
    r"(?P<uuid>[A-F0-9-]+)$"
)


def _clean_stem(name: str) -> str:
    return Path(name).stem


def _load_splits() -> dict[str, str]:
    splits = {}
    for split in ["training", "validation", "test"]:
        for raw_line in (METADATA_DIR / f"{split}.txt").read_text().splitlines():
            line = raw_line.strip()
            if line:
                splits[_clean_stem(line)] = split
    return splits


def _load_plates() -> dict[str, int]:
    plates = {}
    with (METADATA_DIR / "filenames_and_plates.csv").open(newline="") as handle:
        for filename, plate in csv.reader(handle):
            plates[_clean_stem(filename)] = int(plate)
    return plates


def _image_paths() -> list[Path]:
    return sorted(
        p
        for p in IMAGE_DIR.glob("*.tif")
        if not p.name.startswith("._") and p.is_file()
    )


def _mask_path(image_id: str) -> Path:
    path = MASK_DIR / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _decode_mask(mask_path: Path) -> np.ndarray:
    """Convert BBBC039's manual mask into a connected-component label image."""

    rgba = np.array(Image.open(mask_path).convert("RGBA"))
    foreground = rgba[..., 0] > 0
    return label(foreground).astype(np.int32)


def _scale_to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [0.1, 99.9])
    if high <= low:
        high = float(image.max() or 1)
        low = float(image.min())
    scaled = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _metadata_from_name(image_id: str) -> dict[str, str | int]:
    match = IMAGE_RE.match(image_id)
    if not match:
        return {"experiment": "IXMtest", "well": "", "site": 0}
    return {
        "experiment": match.group("experiment"),
        "well": match.group("well"),
        "site": int(match.group("site")),
    }


def _cp_measure_column(group: str, feature: str) -> str:
    if group in {"sizeshape", "feret", "zernike"}:
        if feature.startswith("Center_"):
            return f"Location_{feature}"
        return f"AreaShape_{feature}"
    if group in {"intensity", "radial_distribution", "radial_zernikes"}:
        return f"{feature}_{CHANNEL_NAME}"
    if group == "texture":
        return f"Texture_{feature}_{CHANNEL_NAME}"
    if group == "granularity":
        return f"{feature}_{CHANNEL_NAME}"
    return f"{group}_{feature}_{CHANNEL_NAME}"


@cache
def _expected_cp_measure_columns() -> dict[str, list[str]]:
    masks = np.zeros((20, 20), dtype=np.int32)
    masks[2:8, 3:10] = 1
    masks[10:16, 11:18] = 2
    pixels = np.linspace(0, 1, 400, dtype=np.float32).reshape(20, 20)
    return {
        group: [
            _cp_measure_column(group, feature) for feature in measure(masks, pixels)
        ]
        for group, measure in CP_MEASUREMENTS.items()
    }


def _cp_measure_frame(labels: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    # _measure_objects passes float32 pixels, so the else branch applies:
    # intensities are scaled by the image maximum, not by the dtype maximum.
    pixels = intensity.astype(np.float32)
    if np.issubdtype(intensity.dtype, np.integer):
        pixels = pixels / np.iinfo(intensity.dtype).max
    else:
        high = float(np.nanmax(pixels) or 1)
        if high > 1:
            pixels = pixels / high
    measurements = {}
    object_count = int(labels.max())
    for group, measure in CP_MEASUREMENTS.items():
        try:
            group_measurements = measure(labels, pixels)
        except Exception as exc:
            print(f"  cp_measure {group} failed; filling with NaN ({exc})")
            for column in _expected_cp_measure_columns()[group]:
                measurements[column] = np.full(object_count, np.nan, dtype=np.float32)
            continue
        for feature, values in group_measurements.items():
            measurements[_cp_measure_column(group, feature)] = values
    frame = pd.DataFrame(measurements)
    frame.insert(0, "LocalObjectNumber", np.arange(1, len(frame) + 1))
    return frame


def _measure_objects(
    image_number: int,
    image_id: str,
    labels: np.ndarray,
    intensity: np.ndarray,
    split: str,
    plate: int,
    first_global_id: int,
) -> pd.DataFrame:
    frame = _cp_measure_frame(labels, intensity)
    frame.insert(
        0,
        "ObjectNumber",
        np.arange(first_global_id, first_global_id + len(frame)),
    )
    frame.insert(1, "ImageNumber", image_number)
    frame.insert(2, "image_id", image_id)
    frame.insert(3, "split", split)
    frame.insert(4, "plate", plate)
    frame["object_type"] = "nucleus"
    frame["Location_Center_Z"] = 0.0
    return frame


def _image_row(
    image_number: int,
    image_id: str,
    image_path: Path,
    mask_path: Path,
    image_u16: np.ndarray,
    split: str,
    plate: int,
    object_count: int,
) -> dict[str, object]:
    """Metadata only: the database stores no full images or masks."""

    size_y, size_x = image_u16.shape
    meta = _metadata_from_name(image_id)
    return {
        "ImageNumber": image_number,
        "image_id": image_id,
        "source_filename": image_path.name,
        "mask_filename": mask_path.name,
        "experiment": meta["experiment"],
        "well": meta["well"],
        "site": meta["site"],
        "split": split,
        "plate": plate,
        "object_count": object_count,
        "channel": CHANNEL_NAME,
        "z": 0,
        "height": size_y,
        "width": size_x,
        "dtype": "uint16",
        "size_c": 1,
        "size_z": 1,
        "size_y": size_y,
        "size_x": size_x,
        "source": SOURCE,
        "source_url": SOURCE_URL,
    }


def _crop_rows(
    objects: pd.DataFrame,
    image_u16: np.ndarray,
    image_u8: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, object]]:
    size_y, size_x = image_u16.shape
    rows = []
    for row in objects.itertuples(index=False):
        y0 = max(0, int(row.AreaShape_BoundingBoxMinimum_Y) - CROP_PADDING)
        x0 = max(0, int(row.AreaShape_BoundingBoxMinimum_X) - CROP_PADDING)
        y1 = min(size_y, int(row.AreaShape_BoundingBoxMaximum_Y) + CROP_PADDING)
        x1 = min(size_x, int(row.AreaShape_BoundingBoxMaximum_X) + CROP_PADDING)
        crop_u16 = image_u16[y0:y1, x0:x1]
        crop_u8 = image_u8[y0:y1, x0:x1]
        crop_mask = (labels[y0:y1, x0:x1] == row.LocalObjectNumber).astype(np.uint8)
        rows.append(
            {
                "ImageNumber": int(row.ImageNumber),
                "ObjectNumber": int(row.ObjectNumber),
                "LocalObjectNumber": int(row.LocalObjectNumber),
                "image_id": row.image_id,
                "object_type": "nucleus",
                "channel": CHANNEL_NAME,
                "z": 0,
                "tile_y": y0,
                "tile_x": x0,
                "height": crop_u16.shape[0],
                "width": crop_u16.shape[1],
                "dtype": "uint16",
                "size_c": 1,
                "size_z": 1,
                "size_y": size_y,
                "size_x": size_x,
                "source": SOURCE,
                "source_url": SOURCE_URL,
                "pixel_data_raw": crop_u16.tobytes(),
                "mask_data_raw": crop_mask.tobytes(),
                "pixel_data_jpeg": _encode_jpeg(crop_u8),
            }
        )
    return rows


def _write_representative_preview(image_u8: np.ndarray, labels: np.ndarray) -> None:
    rgb = np.dstack([image_u8, image_u8, image_u8])
    boundary = labels > 0
    for prop in regionprops(label(boundary)):
        y0, x0, y1, x1 = prop.bbox
        rgb[y0:y1, x0] = [121, 194, 208]
        rgb[y0:y1, max(x1 - 1, x0)] = [121, 194, 208]
        rgb[y0, x0:x1] = [121, 194, 208]
        rgb[max(y1 - 1, y0), x0:x1] = [121, 194, 208]
    Image.fromarray(rgb).save(
        PACKAGE_ASSETS_DIR / "example_cell_composite.jpg",
        quality=92,
    )


def main() -> None:
    splits = _load_splits()
    plates = _load_plates()
    image_rows = []
    object_frames = []
    crop_rows = []
    next_object_number = 1

    for image_number, image_path in enumerate(_image_paths(), start=1):
        image_id = image_path.stem
        mask_path = _mask_path(image_id)
        image_u16 = np.array(Image.open(image_path), dtype=np.uint16)
        image_u8 = _scale_to_uint8(image_u16)
        labels = _decode_mask(mask_path)
        split = splits.get(image_id, "unknown")
        plate = plates.get(image_id, 0)
        objects = _measure_objects(
            image_number,
            image_id,
            labels,
            image_u16.astype(np.float32),
            split,
            plate,
            next_object_number,
        )
        next_object_number += len(objects)
        image_rows.append(
            _image_row(
                image_number,
                image_id,
                image_path,
                mask_path,
                image_u16,
                split,
                plate,
                len(objects),
            )
        )
        object_frames.append(objects)
        crop_rows.extend(_crop_rows(objects, image_u16, image_u8, labels))
        if image_id == REPRESENTATIVE_IMAGE_ID:
            _write_representative_preview(image_u8, labels)
            (PACKAGE_ASSETS_DIR / "representative_field.jpg").write_bytes(
                _encode_jpeg(image_u8)
            )
        if image_number % 25 == 0:
            print(f"processed {image_number} images")

    images = pd.DataFrame(image_rows)
    objects = pd.concat(object_frames, ignore_index=True)
    crops = pd.DataFrame(crop_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The package (and so the database) keeps one JPEG crop per nucleus. The
    # full crops stay in output/ for extract_feature_spaces.py (MorphEm).
    package_crops = crops.drop(
        columns=[
            "pixel_data_raw",
            "mask_data_raw",
            "dtype",
        ]
    )
    tables = {
        "bbbc039_images": (images, images),
        "bbbc039_objects": (objects, objects),
        "object_crops": (crops, package_crops),
    }
    for name, (output_frame, package_frame) in tables.items():
        out_path = OUTPUT_DIR / f"{name}.parquet"
        output_frame.to_parquet(out_path, index=False)
        package_frame.to_parquet(PACKAGE_ASSETS_DIR / f"{name}.parquet", index=False)
        print(
            f"wrote {len(output_frame)} rows, {len(output_frame.columns)} columns"
            f" -> {out_path}"
        )


if __name__ == "__main__":
    main()
