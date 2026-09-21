"""Validation checks for JPEG Warehouse artifacts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from picture_thousand_records.duckdb_access import connect_to_warehouse


def validate_artifact(path: str | Path) -> dict[str, Any]:
    """Validate a JPEG Warehouse and return a compact report."""

    jpeg_path = Path(path)
    with Image.open(jpeg_path) as image:
        image.verify()

    with zipfile.ZipFile(jpeg_path) as archive:
        members = archive.namelist()

    manifest_path = jpeg_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    attached = connect_to_warehouse(jpeg_path)
    try:
        con = attached.connection
        counts = {
            "cellprofiler_nuclei": con.sql(
                "SELECT COUNT(*) FROM warehouse.cellprofiler.nuclei"
            ).fetchone()[0],
            "cellprofiler_cells": con.sql(
                "SELECT COUNT(*) FROM warehouse.cellprofiler.cells"
            ).fetchone()[0],
            "cellprofiler_cytoplasm": con.sql(
                "SELECT COUNT(*) FROM warehouse.cellprofiler.cytoplasm"
            ).fetchone()[0],
            "cellprofiler_ph3": con.sql(
                "SELECT COUNT(*) FROM warehouse.cellprofiler.ph3"
            ).fetchone()[0],
            "images_metadata": con.sql(
                "SELECT COUNT(*) FROM warehouse.images.images"
            ).fetchone()[0],
            "images_object_crops": con.sql(
                "SELECT COUNT(*) FROM warehouse.images.object_crops"
            ).fetchone()[0],
            "morphem_features": con.sql(
                "SELECT COUNT(*) FROM warehouse.morphem.features"
            ).fetchone()[0],
            "morphem_knn_graph": con.sql(
                "SELECT COUNT(*) FROM warehouse.morphem.knn_graph"
            ).fetchone()[0],
        }
    finally:
        attached.connection.close()

    return {
        "jpeg": str(jpeg_path),
        "zip_members": members,
        "manifest": manifest,
        "duckdb_access": attached.mode,
        "counts": counts,
    }
