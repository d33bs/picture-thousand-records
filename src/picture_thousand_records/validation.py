"""Validation checks for JPEG Database artifacts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import duckdb
from PIL import Image

from picture_thousand_records.duckdb_access import connect_to_database


def validate_artifact(path: str | Path) -> dict[str, Any]:
    """Validate a JPEG Database and return a compact report."""

    jpeg_path = Path(path)
    with Image.open(jpeg_path) as image:
        image.verify()

    with zipfile.ZipFile(jpeg_path) as archive:
        members = archive.namelist()

    manifest_path = jpeg_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    attached = connect_to_database(jpeg_path)
    try:
        con = attached.connection
        counts = {
            "bbbc039_images": con.sql(
                "SELECT COUNT(*) FROM database.bbbc039.images"
            ).fetchone()[0],
            "bbbc039_objects": con.sql(
                "SELECT COUNT(*) FROM database.bbbc039.objects"
            ).fetchone()[0],
            "bbbc039_annotations": con.sql(
                "SELECT COUNT(*) FROM database.bbbc039.annotations"
            ).fetchone()[0],
            "bbbc039_segmentations": con.sql(
                "SELECT COUNT(*) FROM database.bbbc039.segmentations"
            ).fetchone()[0],
            "images_metadata": con.sql(
                "SELECT COUNT(*) FROM database.images.images"
            ).fetchone()[0],
            "images_object_crops": con.sql(
                "SELECT COUNT(*) FROM database.images.object_crops"
            ).fetchone()[0],
            "features_cp_measure": con.sql(
                "SELECT COUNT(*) FROM database.features.cp_measure"
            ).fetchone()[0],
            "features_morphem": con.sql(
                "SELECT COUNT(*) FROM database.features.morphem"
            ).fetchone()[0],
            "features_fused": con.sql(
                "SELECT COUNT(*) FROM database.features.fused"
            ).fetchone()[0],
            "features_knn_graph": con.sql(
                "SELECT COUNT(*) FROM database.features.knn_graph"
            ).fetchone()[0],
        }
        index_scan_used = attached.hnsw and _uses_hnsw_index(con)
    finally:
        attached.connection.close()

    return {
        "jpeg": str(jpeg_path),
        "zip_members": members,
        "manifest": manifest,
        "duckdb_access": attached.mode,
        "hnsw_loaded": attached.hnsw,
        "hnsw_index_scan_used": index_scan_used,
        "counts": counts,
    }


def _uses_hnsw_index(con: duckdb.DuckDBPyConnection) -> bool:
    """Make sure that a nearest-neighbor query plans an HNSW index scan."""

    con.execute(
        "SET VARIABLE query_vector = (SELECT fused_embedding "
        "FROM database.features.fused WHERE ObjectNumber = 1)"
    )
    plan = con.execute(
        "EXPLAIN SELECT ObjectNumber FROM database.features.fused "
        "ORDER BY array_distance(fused_embedding, "
        "getvariable('query_vector')::FLOAT[48]) LIMIT 13"
    ).fetchall()
    return "HNSW_INDEX_SCAN" in " ".join(str(cell) for row in plan for cell in row)
