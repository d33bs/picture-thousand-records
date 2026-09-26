"""Build the JPEG Database polyglot artifact.

The database uses the full BBBC039 benchmark: 200 U2OS DNA-channel
microscopy fields, manual nucleus annotations, per-nucleus crops,
cp_measure measurements, MorphEm embeddings, fused features, 3-D PCA, UMAP,
and t-SNE maps
coordinates, and DuckDB HNSW indexes (the `vss` extension) that answer the
nearest-neighbor searches.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from PIL import Image, ImageDraw, ImageFont

from picture_thousand_records.diagrams import all_diagrams

ASSETS_DIR = Path(__file__).parent / "assets"
SOURCE_URL = "https://bbbc.broadinstitute.org/BBBC039"
REPO_URL = "https://github.com/d33bs/picture-thousand-records"
# The two accent colors of the logo, sampled from its pixels (the mean color of
# the "picture" and "records" letters), so the page title matches the cover.
LOGO_BLUE = "#03a0f1"
LOGO_PINK = "#e1096f"
MISSED_COLOR = "#c77dff"
MORPHEM_MODEL = "CaicedoLab/MorphEm"
CP_MEASURE_DIM = 16
# cp_measure runs at data-preparation time (tools/bbbc039_data), not at build
# time, so this pin mirrors the version in uv.lock (a test keeps them equal).
CP_MEASURE_VERSION = "0.2.0"
CP_MEASURE_URL = "https://github.com/afermg/cp_measure"
CP_MEASURE_PAPER_URL = "https://arxiv.org/abs/2507.01163"
# Cellpose also runs at data-preparation time
# (tools/bbbc039_data/match_cellpose.py), not at build time, so this pin
# mirrors the version in uv.lock (a test keeps them equal).
CELLPOSE_VERSION = "4.2.1.1"
CELLPOSE_URL = "https://github.com/MouseLand/cellpose"
CELLPOSE_MODEL = "cpsam_v2"
CELLPOSE_IOU_THRESHOLD = 0.5
MORPHEM_EMBEDDING_DIM = 384
FUSED_EMBEDDING_DIM = 48
DEFAULT_FEATURE_SPACE = "fused"
# Neighbor-density colors, sparse to dense. Saturated stops keep the ramp from
# passing through gray, and blue-green-yellow stays readable with the common
# kinds of color blindness. The page's dots and legend both use these.
DENSITY_STOPS = [(88, 130, 255), (72, 205, 150), (255, 224, 64)]
KNN_GRAPH_K = 6
KNN_INSERT_BATCH = 2000
RECALL_SAMPLE_STEP = 100

# HNSW indexes are built by DuckDB's `vss` extension. The `l2sq` metric is the
# one DuckDB's optimizer pairs with `array_distance`.
HNSW_OPTIONS = "metric = 'l2sq', M = 16, ef_construction = 200, ef_search = 100"


@dataclass(frozen=True)
class FeatureSpace:
    """One feature table and whether the database stores an HNSW index for it."""

    name: str
    table: str
    column: str
    dim: int
    asset: str
    indexed: bool


FEATURE_TABLES = [
    FeatureSpace(
        "cp_measure",
        "features.cp_measure",
        "measurement_vector",
        CP_MEASURE_DIM,
        "features_cp_measure.parquet",
        True,
    ),
    # The 384-d MorphEm index would add about 33 MB, as much as the vectors
    # themselves, so MorphEm searches scan all vectors instead. That is still
    # instant for 19,565 nuclei.
    FeatureSpace(
        "morphem",
        "features.morphem",
        "embedding",
        MORPHEM_EMBEDDING_DIM,
        "features_morphem.parquet",
        False,
    ),
    FeatureSpace(
        "fused",
        "features.fused",
        "fused_embedding",
        FUSED_EMBEDDING_DIM,
        "features_fused.parquet",
        True,
    ),
]
INDEXED_SPACES = [space for space in FEATURE_TABLES if space.indexed]


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths produced by a database build."""

    output_dir: Path
    jpeg: Path
    duckdb: Path
    manifest: Path
    readme: Path
    browser_page: Path


def build_artifacts(output_dir: str | Path = ".") -> ArtifactPaths:
    """Build a JPEG/ZIP polyglot containing a BBBC039 DuckDB database."""

    paths = _paths(Path(output_dir))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _unlink_outputs(paths)
    counts = _build_duckdb(paths.duckdb)
    manifest = _manifest(counts)
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths.readme.write_text(_embedded_readme(counts), encoding="utf-8")
    _write_cover_jpeg(paths.jpeg)
    _append_zip(paths)
    counts["jpeg_bytes"] = paths.jpeg.stat().st_size
    _write_browser_page(paths.browser_page, counts)
    return paths


def _paths(output_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        output_dir=output_dir,
        jpeg=output_dir / "database.jpg",
        duckdb=output_dir / "database.duckdb",
        manifest=output_dir / "manifest.json",
        readme=output_dir / "DATABASE_README.md",
        browser_page=output_dir / "index.html",
    )


def _unlink_outputs(paths: ArtifactPaths) -> None:
    for path in [
        paths.jpeg,
        paths.duckdb,
        paths.manifest,
        paths.readme,
        paths.browser_page,
    ]:
        if path.exists():
            path.unlink()


def _build_duckdb(path: Path) -> dict[str, int]:
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA bbbc039")
        con.execute("CREATE SCHEMA images")
        con.execute("CREATE SCHEMA features")

        counts = {
            "bbbc039_images": _load_parquet_table(
                con, "bbbc039_images.parquet", "bbbc039.images"
            ),
            "bbbc039_objects": _load_objects_table(con),
        }
        _create_bbbc039_views(con)
        counts["images_metadata"] = _load_parquet_table(
            con, "bbbc039_images.parquet", "images.images"
        )
        counts["images_object_crops"] = _load_parquet_table(
            con, "object_crops.parquet", "images.object_crops"
        )
        counts["features_cp_measure"] = _create_vector_table(
            con,
            "features_cp_measure.parquet",
            "features.cp_measure",
            "measurement_vector",
            CP_MEASURE_DIM,
        )
        counts["features_morphem"] = _create_vector_table(
            con,
            "features_morphem.parquet",
            "features.morphem",
            "embedding",
            MORPHEM_EMBEDDING_DIM,
        )
        counts["features_fused"] = _create_vector_table(
            con,
            "features_fused.parquet",
            "features.fused",
            "fused_embedding",
            FUSED_EMBEDDING_DIM,
        )
        counts["features_embedding_quality"] = _load_parquet_table(
            con, "embedding_quality.parquet", "features.embedding_quality"
        )
        _create_hnsw_indexes(con)
        counts["features_knn_graph"] = _create_knn_graph(con)
        counts["hnsw_recall"] = _measure_hnsw_recall(con)
        counts.update(_dataset_stats(con))
        return counts
    finally:
        con.close()


def load_vss(con: duckdb.DuckDBPyConnection) -> None:
    """Load DuckDB's `vss` extension (needs network on first use)."""

    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    # HNSW indexes are stored in the database file only with this flag.
    con.execute("SET hnsw_enable_experimental_persistence = true")


def _create_hnsw_indexes(con: duckdb.DuckDBPyConnection) -> None:
    load_vss(con)
    for space in INDEXED_SPACES:
        con.execute(
            f"CREATE INDEX {space.name}_hnsw ON {space.table} "
            f"USING HNSW ({space.column}) WITH ({HNSW_OPTIONS})"
        )


def _vector_literal(vector: list[float], dim: int) -> str:
    # The optimizer only uses an HNSW index when the query vector is a constant.
    return "[" + ",".join(repr(float(value)) for value in vector) + f"]::FLOAT[{dim}]"


def _create_knn_graph(con: duckdb.DuckDBPyConnection) -> int:
    """Store each nucleus's nearest neighbors, found through HNSW indexes.

    The neighbors come from temporary in-memory copies of the vectors with the
    same index settings. That way the MorphEm index, which the database does
    not keep, still answers the queries quickly. The page only uses this table
    to color the map by neighbor density.
    """

    con.execute(
        """
        CREATE TABLE features.knn_graph (
            feature_space VARCHAR,
            ObjectNumber BIGINT,
            neighbor_object_number BIGINT,
            rank BIGINT,
            distance DOUBLE
        )
        """
    )
    scratch = duckdb.connect()
    try:
        load_vss(scratch)
        for space in FEATURE_TABLES:
            asset_path = (ASSETS_DIR / space.asset).as_posix()
            scratch.execute(
                f"CREATE OR REPLACE TABLE t AS SELECT ObjectNumber, "
                f"CAST({space.column} AS FLOAT[{space.dim}]) AS v "
                f"FROM read_parquet('{asset_path}') ORDER BY ObjectNumber"
            )
            scratch.execute(
                f"CREATE INDEX t_hnsw ON t USING HNSW (v) WITH ({HNSW_OPTIONS})"
            )
            _insert_neighbors(con, scratch, space)
            scratch.execute("DROP TABLE t")
    finally:
        scratch.close()
    return con.sql("SELECT COUNT(*) FROM features.knn_graph").fetchone()[0]


def _insert_neighbors(
    con: duckdb.DuckDBPyConnection,
    scratch: duckdb.DuckDBPyConnection,
    space: FeatureSpace,
) -> None:
    rows: list[str] = []
    for object_number, vector in scratch.sql(
        "SELECT ObjectNumber, v FROM t ORDER BY ObjectNumber"
    ).fetchall():
        hits = scratch.execute(
            f"SELECT ObjectNumber, array_distance(v, "
            f"{_vector_literal(vector, space.dim)}) AS d FROM t "
            f"ORDER BY d LIMIT {KNN_GRAPH_K + 1}"
        ).fetchall()
        neighbors = [hit for hit in hits if hit[0] != object_number]
        for rank, (neighbor, distance) in enumerate(neighbors[:KNN_GRAPH_K], start=1):
            rows.append(
                f"('{space.name}', {object_number}, {neighbor}, {rank}, {distance!r})"
            )
        if len(rows) >= KNN_INSERT_BATCH:
            con.execute("INSERT INTO features.knn_graph VALUES " + ",".join(rows))
            rows = []
    if rows:
        con.execute("INSERT INTO features.knn_graph VALUES " + ",".join(rows))


def _measure_hnsw_recall(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """Fraction of the exact 12 nearest neighbors the HNSW index also returns.

    Checked on every 100th nucleus (about 200) in each indexed feature space,
    against an exact scan with the index turned off.
    """

    recall: dict[str, float] = {}
    for space in INDEXED_SPACES:
        table, column, dim = space.table, space.column, space.dim
        sampled = con.sql(
            f"SELECT ObjectNumber, {column} FROM {table} "
            f"WHERE ObjectNumber % {RECALL_SAMPLE_STEP} = 0 ORDER BY ObjectNumber"
        ).fetchall()
        found = total = 0
        for object_number, vector in sampled:
            query = (
                f"SELECT ObjectNumber FROM {table} WHERE ObjectNumber != "
                f"{object_number} ORDER BY array_distance({column}, "
                f"{_vector_literal(vector, dim)}) LIMIT 12"
            )
            # The exact scan needs a plan without the index; the indexed
            # query must take the HNSW path (ORDER BY ... LIMIT only).
            con.execute("SET disabled_optimizers = 'extension'")
            exact = {row[0] for row in con.execute(query).fetchall()}
            con.execute("RESET disabled_optimizers")
            indexed_query = (
                f"SELECT ObjectNumber FROM {table} ORDER BY array_distance("
                f"{column}, {_vector_literal(vector, dim)}) LIMIT 13"
            )
            indexed = {row[0] for row in con.execute(indexed_query).fetchall()} - {
                object_number
            }
            found += len(exact & indexed)
            total += len(exact)
        recall[space.name] = round(found / total, 4)
    return recall


def _dataset_stats(con: duckdb.DuckDBPyConnection) -> dict[str, int | float]:
    train, validation, test = con.sql(
        """
        SELECT
            SUM((split = 'training')::INT),
            SUM((split = 'validation')::INT),
            SUM((split = 'test')::INT)
        FROM bbbc039.images
        """
    ).fetchone()
    median_nuclei, max_nuclei = con.sql(
        """
        SELECT median(object_count), max(object_count)
        FROM bbbc039.images
        """
    ).fetchone()
    captured, total = con.sql(
        """
        SELECT COUNT(*) FILTER (WHERE cellpose_captured), COUNT(*)
        FROM bbbc039.objects
        """
    ).fetchone()
    return {
        "training_images": train,
        "validation_images": validation,
        "test_images": test,
        "median_nuclei_per_image": round(median_nuclei),
        "max_nuclei_per_image": max_nuclei,
        "cellpose_captured": captured,
        "cellpose_capture_pct": round(100 * captured / total, 1),
    }


def _load_objects_table(con: duckdb.DuckDBPyConnection) -> int:
    """Load `bbbc039.objects`, storing the DOUBLE measurements as 32-bit FLOAT.

    That cuts the table from about 56 MB to about 21 MB. Single precision keeps
    about 7 significant digits, which is far more than these measurements need.

    The Cellpose capture columns come from a separate asset (written by
    tools/bbbc039_data/match_cellpose.py) joined in here, so the base
    extraction does not need the torch stack Cellpose pulls in.
    """

    objects_path = (ASSETS_DIR / "bbbc039_objects.parquet").as_posix()
    cellpose_path = (ASSETS_DIR / "bbbc039_cellpose_matches.parquet").as_posix()
    columns = con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{objects_path}')"
    ).fetchall()
    doubles = ", ".join(
        f'"{name}"::FLOAT AS "{name}"' for name, kind, *_ in columns if kind == "DOUBLE"
    )
    con.execute(
        f"CREATE TABLE bbbc039.objects AS "
        f"SELECT o.* REPLACE ({doubles}), "
        f"c.cellpose_captured, "
        f"CAST(c.cellpose_iou AS FLOAT) AS cellpose_iou "
        f"FROM read_parquet('{objects_path}') o "
        f"LEFT JOIN read_parquet('{cellpose_path}') c USING (ObjectNumber)"
    )
    missing = con.sql(
        "SELECT COUNT(*) FROM bbbc039.objects "
        "WHERE cellpose_captured IS NULL OR cellpose_iou IS NULL"
    ).fetchone()[0]
    if missing:
        raise RuntimeError(
            f"{missing} nuclei have no Cellpose capture verdict. The "
            "bbbc039_cellpose_matches.parquet asset is missing or stale "
            "relative to bbbc039_objects.parquet; re-run "
            "tools/bbbc039_data/match_cellpose.py "
            "(uv run --group real_data --group cellpose) and rebuild."
        )
    return con.sql("SELECT COUNT(*) FROM bbbc039.objects").fetchone()[0]


def _load_parquet_table(
    con: duckdb.DuckDBPyConnection,
    asset_name: str,
    table_name: str,
) -> int:
    asset_path = ASSETS_DIR / asset_name
    con.execute(
        f"CREATE TABLE {table_name} AS "
        f"SELECT * FROM read_parquet('{asset_path.as_posix()}')"
    )
    return con.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def _create_bbbc039_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE VIEW bbbc039.annotations AS SELECT * FROM bbbc039.objects")
    con.execute(
        """
        CREATE VIEW bbbc039.segmentations AS
        SELECT
            ImageNumber,
            image_id,
            split,
            plate,
            object_count AS annotated_nuclei,
            'manual nuclei masks' AS annotation_type,
            mask_filename
        FROM bbbc039.images
        """
    )


def _create_vector_table(
    con: duckdb.DuckDBPyConnection,
    asset_name: str,
    table_name: str,
    vector_column: str,
    vector_dim: int,
) -> int:
    asset_path = ASSETS_DIR / asset_name
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            * EXCLUDE ({vector_column}),
            CAST({vector_column} AS FLOAT[{vector_dim}]) AS {vector_column}
        FROM read_parquet('{asset_path.as_posix()}')
        """
    )
    return con.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def _manifest(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "format": "jpeg-database",
        "version": "0.2",
        "title": "BBBC039 JPEG Database",
        "database": "database.duckdb",
        "database_engine": "duckdb",
        "access": {
            "filesystem": "zipfs",
            "member": "database.duckdb",
            "split": "!!",
        },
        "schemas": ["bbbc039", "images", "features"],
        "rows": {
            "bbbc039.images": counts["bbbc039_images"],
            "bbbc039.objects": counts["bbbc039_objects"],
            "bbbc039.annotations": counts["bbbc039_objects"],
            "bbbc039.segmentations": counts["bbbc039_images"],
            "images.images": counts["images_metadata"],
            "images.object_crops": counts["images_object_crops"],
            "features.cp_measure": counts["features_cp_measure"],
            "features.morphem": counts["features_morphem"],
            "features.fused": counts["features_fused"],
            "features.knn_graph": counts["features_knn_graph"],
            "features.embedding_quality": counts["features_embedding_quality"],
        },
        "source": {
            "dataset": "Broad Bioimage Benchmark Collection BBBC039",
            "source_url": SOURCE_URL,
            "cell_type": "human U2OS cells",
            "channel": "DNA / Hoechst",
            "image_count": counts["bbbc039_images"],
            "annotated_nuclei": counts["bbbc039_objects"],
            "manual_annotations": "BBBC039 PNG masks decoded to nucleus objects",
            "feature_spaces": ["cp_measure", "morphem", "fused"],
            "measured_with": (
                "cp_measure (CellProfiler measurements) on the manual masks; "
                "MorphEm embeddings on the DNA crops; scikit-image to decode "
                "the masks. The database does not store the original images "
                "or masks; download them from BBBC039 to re-measure"
            ),
            "pipeline": "tools/bbbc039_data",
            "stored_images": "one 8-bit JPEG crop per nucleus (images.object_crops)",
            "not_stored": [
                "original TIFF images",
                "manual mask PNGs",
                "16-bit raw crops and crop masks",
                "JPEG XL crops",
                "Cellpose masks",
            ],
            "crop_columns": ["pixel_data_jpeg"],
        },
        "cellpose": {
            "library": "cellpose",
            "library_version": CELLPOSE_VERSION,
            "library_url": CELLPOSE_URL,
            "model": CELLPOSE_MODEL,
            "diameter": "auto-estimated per image",
            "min_size": 40,
            "iou_threshold": CELLPOSE_IOU_THRESHOLD,
            "matching": (
                "one-to-one greedy by IoU (best pair first): a manual nucleus "
                "is captured when a Cellpose mask overlaps it with IoU >= 0.5, "
                "and each Cellpose mask captures at most one manual nucleus"
            ),
            "columns": ["cellpose_captured", "cellpose_iou"],
            "captured_nuclei": counts["cellpose_captured"],
            "missed_nuclei": counts["bbbc039_objects"] - counts["cellpose_captured"],
            "captured_pct": counts["cellpose_capture_pct"],
            "note": (
                "cellpose_iou holds the matched pair's IoU for captured "
                "nuclei and the best IoU reached with any Cellpose mask "
                "otherwise. Cellpose masks are not stored; re-run "
                "tools/bbbc039_data/match_cellpose.py to reproduce them"
            ),
        },
        "features": {
            "default_feature_space": DEFAULT_FEATURE_SPACE,
            "cp_measure": {
                "library": "cp_measure",
                "library_version": CP_MEASURE_VERSION,
                "library_url": CP_MEASURE_URL,
                "reference": "Munoz et al. (2025), arXiv:2507.01163",
                "call": "cp_measure.bulk.get_core_measurements(legacy=True)",
                "input": (
                    "BBBC039 manual nucleus masks and the DNA channel, "
                    "scaled to 0-1 by each image's maximum"
                ),
                "raw_table": "bbbc039.objects",
                "measurements_per_nucleus": 271,
                "embedding_dim": CP_MEASURE_DIM,
                "vector_column": "measurement_vector",
                "description": (
                    "PCA reduction (16 dimensions) of the 263 AreaShape, "
                    "Granularity, Intensity, RadialDistribution, and Texture "
                    "columns that cp_measure computed; the full measurements "
                    "are in bbbc039.objects"
                ),
            },
            "morphem": {
                "model": MORPHEM_MODEL,
                "model_url": f"https://huggingface.co/{MORPHEM_MODEL}",
                "embedding_dim": MORPHEM_EMBEDDING_DIM,
                "vector_column": "embedding",
                "channels": ["DNA"],
            },
            "fused": {
                "embedding_dim": FUSED_EMBEDDING_DIM,
                "vector_column": "fused_embedding",
                "description": (
                    "cp_measure PCA features concatenated with MorphEm PCA features"
                ),
            },
            "maps": {
                "n_components": 3,
                "note": (
                    "Each feature table has three 3-D maps of the same vectors. "
                    "The page draws all three side by side and marks the "
                    "HNSW search result on each."
                ),
                "pca": {
                    "method": "PCA",
                    "library": "scikit-learn",
                    "reference": "Jolliffe & Cadima (2016)",
                    "random_state": 42,
                    "columns": ["pca_x", "pca_y", "pca_z"],
                },
                "umap": {
                    "method": "UMAP",
                    "library": "umap-learn",
                    "reference": "McInnes, Healy & Melville (2018)",
                    "metric": "euclidean",
                    "random_state": 42,
                    "columns": ["umap_x", "umap_y", "umap_z"],
                },
                "tsne": {
                    "method": "t-SNE (Barnes-Hut, 3-D)",
                    "library": "openTSNE",
                    "reference": "Policar, Strazar & Zupan (2024); "
                    "van der Maaten & Hinton (2008)",
                    "perplexity": 30,
                    "metric": "euclidean",
                    "random_state": 42,
                    "columns": ["tsne_x", "tsne_y", "tsne_z"],
                },
                "quality_table": "features.embedding_quality",
                "quality_measure": (
                    "neighbor_recall: share of a nucleus's 6 exact nearest "
                    "neighbors (in the feature space) that are also among its 15 "
                    "nearest points on the map. distance_ratio: median map "
                    "distance to a true neighbor over the median distance "
                    "between random pairs (lower is better)."
                ),
            },
            "hnsw_index": {
                "method": "HNSW index (used by the page's nearest-neighbor search)",
                "note": (
                    "Stored for cp_measure and fused. MorphEm has no stored index "
                    "(it would add about 33 MB), so MorphEm searches scan all "
                    "vectors with the same SQL."
                ),
                "extension": "vss",
                "extension_url": "https://duckdb.org/docs/current/core_extensions/vss",
                "reference": "Malkov & Yashunin (2016)",
                "metric": "l2sq (paired with array_distance)",
                "options": HNSW_OPTIONS,
                "indexes": {
                    space.name: f"{space.table}({space.column})"
                    for space in INDEXED_SPACES
                },
                "not_indexed": [
                    space.name for space in FEATURE_TABLES if not space.indexed
                ],
                "recall_at_12_vs_exact_scan": counts["hnsw_recall"],
                "recall_sample": f"every {RECALL_SAMPLE_STEP}th nucleus",
                "load": [
                    "INSTALL vss",
                    "LOAD vss",
                    "SET hnsw_enable_experimental_persistence = true",
                ],
            },
            "knn_graph": {
                "description": (
                    "k=6 neighbors of every nucleus in all three feature "
                    "spaces, found with HNSW indexes (temporary for MorphEm); "
                    "the page averages the 6 distances into a neighbor-density color"
                ),
                "k": KNN_GRAPH_K,
                "edge_table": "features.knn_graph",
            },
        },
        "views": {
            "bbbc039.annotations": "manual nucleus objects decoded from BBBC039 masks",
            "bbbc039.segmentations": "one row per manually annotated image mask",
        },
    }


def _embedded_readme(counts: dict[str, int]) -> str:
    return f"""# JPEG Database

This JPEG contains an embedded ZIP archive with a DuckDB database.

DuckDB access (run `duckdb` in the folder that holds `database.jpg`):

```sql
INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';

-- ATTACH cannot open a zip:// path directly (DuckDB 1.5.x), so stream the
-- stored member out with read_blob, then attach the copy.
COPY (SELECT content FROM read_blob('zip://database.jpg!!database.duckdb'))
    TO 'database.duckdb' (FORMAT blob);

-- The database holds HNSW indexes. Load the vss extension to use them.
INSTALL vss;
LOAD vss;
SET hnsw_enable_experimental_persistence = true;

ATTACH 'database.duckdb' AS database (READ_ONLY);
```

To keep everything in memory afterwards, copy the tables and drop the file:

```sql
CREATE SCHEMA bbbc039; CREATE SCHEMA images; CREATE SCHEMA features;
CREATE TABLE bbbc039.objects AS SELECT * FROM database.bbbc039.objects;
CREATE TABLE features.fused AS SELECT * FROM database.features.fused;
-- ...repeat for any other table you need, then:
DETACH database;
.shell rm database.duckdb
```

`CREATE TABLE ... AS SELECT` copies the data but not the HNSW indexes. To
search with an index, query the attached `database` catalog.

## What is inside this file

`database.jpg` is two files joined end to end. The first part is a JPEG
picture. The second part is a ZIP archive. The
archive holds two files:

- `database.duckdb`: the database. DuckDB (a program that keeps tables in
  one file) reads it.
- `README.md`: this file.

A JPEG ends with the two-byte marker `FF D9`, which means that the picture
ends here. A picture viewer stops at that marker. A ZIP archive keeps its
table of contents at the end of the file, so a ZIP tool reads the end first.
Each tool sees only its own part.

Every table comes from the public BBBC039 benchmark. BBBC039 has 200 DNA
microscopy fields of human U2OS cells. The source package includes manual
nucleus masks for segmentation research.

```text
bbbc039.images          {counts["bbbc039_images"]} rows, one field of view each
bbbc039.objects         {counts["bbbc039_objects"]} rows, one manual nucleus each
bbbc039.annotations     view over the same manual objects
bbbc039.segmentations   view with one row per annotated mask
images.images           {counts["images_metadata"]} rows, image metadata (no pixels)
images.object_crops     {counts["images_object_crops"]} rows, one JPEG crop per nucleus
features.cp_measure     {counts["features_cp_measure"]} cp_measure PCA vectors (16-d)
features.morphem        {counts["features_morphem"]} MorphEm embedding vectors
features.fused          {counts["features_fused"]} fused feature vectors
features.knn_graph      {counts["features_knn_graph"]} neighbor edges (k=6)
```

`bbbc039.objects` stores one row per manual nucleus. It includes image IDs,
split labels, plate numbers, and 271 measurements per nucleus. The
[cp_measure](https://github.com/afermg/cp_measure) library (version
0.2.0) computed the measurements. cp_measure is a Python library that
computes the same kinds of measurements as CellProfiler: size, shape,
DNA intensity, texture, granularity, and radial distribution. We ran it on
the BBBC039 manual masks with `get_core_measurements(legacy=True)`, which
uses CellProfiler's original percentile convention. Each image was scaled
to 0-1 by its maximum first, so intensity values are relative to the
brightest pixel of that image. Some values are NaN where a measurement is
undefined (for example the three `AreaShape_NormalizedMoment` columns).

Two more columns record how Cellpose fared against these manual annotations.
The [cellpose]({CELLPOSE_URL}) library (version {CELLPOSE_VERSION}, model
`{CELLPOSE_MODEL}`) segmented the same DNA fields, and a manual nucleus counts
as captured (`cellpose_captured = true`) when a Cellpose mask overlapped it
with IoU >= 0.5 under one-to-one matching. Cellpose captured
{counts["cellpose_capture_pct"]}% of the {counts["bbbc039_objects"]:,} manually
annotated nuclei. `cellpose_iou` holds the matched pair's IoU for captured
nuclei and the best IoU reached with any mask otherwise. The Cellpose masks
themselves are not stored; re-run `tools/bbbc039_data/match_cellpose.py` to
reproduce them.
IoU means intersection over union: the shared area divided by the total area
covered by either mask. It is also called the
[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index). An IoU of 1
means perfect overlap, and 0 means no overlap.

The page overlays the {counts["bbbc039_objects"] - counts["cellpose_captured"]:,}
manual nuclei Cellpose missed on the PCA, UMAP, and t-SNE maps. Those maps
were computed from manual masks and DNA crops; the Cellpose verdict was not
part of the feature vector. If missed nuclei cluster, that suggests they share
measured shape, intensity, or learned visual neighborhoods. If they scatter,
the miss may depend on image context, touching cells, or segmentation behavior
that these nucleus-level features do not capture. cp_measure is easiest to
interpret through explicit size, shape, brightness, texture, and radial
intensity columns; MorphEm can group subtler crop appearance but is harder to
explain; Fused combines both, but you still need to query the source columns
and crops to decide which side explains a miss.

```sql
SELECT
    ObjectNumber,
    image_id,
    split,
    ROUND(AreaShape_Area, 1) AS area,
    ROUND(Intensity_MeanIntensity_DNA, 4) AS mean_dna,
    ROUND(Texture_Contrast_3_00_256_DNA, 3) AS texture_contrast,
    ROUND(Granularity_1_DNA, 3) AS granularity_1,
    ROUND(RadialDistribution_MeanFrac_1of4_DNA, 3) AS radial_mean_1of4
FROM database.bbbc039.objects
LIMIT 10;
```

The database does not store the original TIFF images or the manual mask
PNGs. To measure the nuclei again, download them from
[BBBC039]({SOURCE_URL}). The only pixels in the database are the JPEG crops.

Each `images.object_crops` row is a self-describing record: alongside the
crop's own `tile_y`/`tile_x`/`height`/`width`, it also carries the source
field's own `size_c`/`size_z`/`size_y`/`size_x` and `source`/`source_url`, so
no join to `images.images` is required. The row stores one 8-bit JPEG crop of
the nucleus in `pixel_data_jpeg`. The crop is scaled for display, so it does
not hold the original 16-bit intensities.

Example query:

```sql
SELECT ObjectNumber, image_id, size_y, size_x, height, width,
       octet_length(pixel_data_jpeg) AS jpeg_bytes
FROM database.images.object_crops
LIMIT 10;
```

The database has three feature spaces. `features.cp_measure` holds a
16-dimension PCA reduction of 263 of the cp_measure measurements in
`bbbc039.objects` (the AreaShape, Granularity, Intensity,
RadialDistribution, and Texture columns). `features.morphem` uses the
pretrained [{MORPHEM_MODEL}](https://huggingface.co/{MORPHEM_MODEL}) model on
the DNA crop. `features.fused` joins the two reduced vectors.

`features.cp_measure` and `features.fused` have an HNSW index on their vector
column, built with DuckDB's `vss` extension (metric `l2sq`). The page's search
uses these indexes. `features.morphem` has no stored index: the 384-dimension
index would add about 33 MB, so MorphEm searches scan all 19,565 vectors with
the same SQL, which is still instant. DuckDB uses an index only when the query
vector is a constant, so copy the seed vector into a variable first:

```sql
SET VARIABLE query_vector = (
    SELECT fused_embedding FROM database.features.fused WHERE ObjectNumber = 1
);

SELECT ObjectNumber,
       array_distance(fused_embedding, getvariable('query_vector')::FLOAT[48])
           AS distance
FROM database.features.fused
ORDER BY distance
LIMIT 10;
```

Run `EXPLAIN` on the query to see `HNSW_INDEX_SCAN`. If `vss` is not loaded,
the same query still works as an exact scan. The manifest records how well the
two indexes agree with an exact scan.

`bbbc039.objects` stores the measurements as 32-bit `FLOAT` (about 7
significant digits), which cuts that table from about 56 MB to about 21 MB.

Each feature table also has three 3-D maps of its vectors: `pca_x`/`pca_y`/
`pca_z` (PCA), `umap_x`/`umap_y`/`umap_z` (UMAP), and `tsne_x`/`tsne_y`/`tsne_z`
(t-SNE, from openTSNE). The page draws all three as point clouds that you
rotate and zoom. `features.embedding_quality` records how well each map keeps
the true nearest neighbors of every nucleus close together. The table
`features.knn_graph` stores the 6 nearest neighbors of every nucleus in all
three feature spaces. The build found them through the HNSW indexes.

```sql
SELECT ObjectNumber, neighbor_object_number, rank, distance
FROM database.features.knn_graph
WHERE feature_space = 'fused' AND ObjectNumber = 1
ORDER BY rank;
```
"""


def _write_cover_jpeg(path: Path) -> None:
    width, height = 1440, 1440
    image = Image.new("RGB", (width, height), "#f6f7f2")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()

    grid_top, grid_bottom, mid_x = 0, height, width // 2
    mid_y = grid_top + (grid_bottom - grid_top) // 2
    quadrants = [
        (0, grid_top, mid_x, mid_y),
        (mid_x, grid_top, width, mid_y),
        (0, mid_y, mid_x, grid_bottom),
        (mid_x, mid_y, width, grid_bottom),
    ]
    for x0, y0, x1, y1 in quadrants:
        draw.rectangle((x0, y0, x1, y1), outline="#d8ddd2", width=2)

    _draw_logo_panel(image, quadrants[0])
    _draw_source_image_panel(image, quadrants[1])
    _draw_schema_panel(draw, fonts, quadrants[2])
    _draw_guide_panel(draw, fonts, quadrants[3])

    image.save(path, format="JPEG", quality=92, optimize=True)


def _load_cover_source_image() -> Image.Image:
    # IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB, scaled to 8 bits.
    # Only this one preview is kept; the database stores no full images.
    return Image.open(ASSETS_DIR / "representative_field.jpg").convert("RGB")


def _draw_logo_panel(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    logo_path = ASSETS_DIR / "picture-thousand-records.png"
    logo = Image.open(logo_path).convert("RGB")
    padding = 6
    target_w = (x1 - x0) - 2 * padding
    target_h = (y1 - y0) - 2 * padding
    fit = min(target_w / logo.width, target_h / logo.height)
    resized = logo.resize(
        (max(1, int(logo.width * fit)), max(1, int(logo.height * fit))),
        Image.LANCZOS,
    )
    paste_x = x0 + ((x1 - x0) - resized.width) // 2
    paste_y = y0 + ((y1 - y0) - resized.height) // 2
    image.paste(resized, (paste_x, paste_y))


def _draw_source_image_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    padding = 4
    target_w = (x1 - x0) - 2 * padding
    target_h = (y1 - y0) - 2 * padding
    source = _load_cover_source_image()
    fit = max(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (max(1, int(source.width * fit)), max(1, int(source.height * fit))),
        Image.LANCZOS,
    )
    crop_x = max(0, (resized.width - target_w) // 2)
    crop_y = max(0, (resized.height - target_h) // 2)
    cropped = resized.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    image.paste(cropped, (x0 + padding, y0 + padding))


# The cover's schema tree. The file holds a picture and, after it, a DuckDB
# database. Its tables sit in three schemas, drawn in two columns.
SCHEMA_TREE_COL_A = [
    ("bbbc039", ["images", "objects", "annotations", "segmentations"]),
    ("images", ["images", "object_crops"]),
]
SCHEMA_TREE_COL_B = [
    (
        "features",
        ["cp_measure", "morphem", "fused", "knn_graph", "embedding_quality"],
    ),
]
TREE_LINE_COLOR = "#3c7d89"


def _draw_schema_panel(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0 + 40, y0 + 24), "Schema", font=fonts["heading"], fill="#20303c")

    font = fonts["mono_schema"]
    size = getattr(font, "size", 24)
    char_width = draw.textlength("M", font=font)
    root_x, root_y = x0 + 40, y0 + 84

    # Fit every line above the caption: 3 lines for the file and its two
    # parts, one connector gap, then the taller of the two table columns.
    rows = max(
        sum(1 + len(tables) for _, tables in column) + len(column) // 2
        for column in (SCHEMA_TREE_COL_A, SCHEMA_TREE_COL_B)
    )
    caption_top = y1 - CAPTION_HEIGHT
    line = min(size + 10, (caption_top - 14 - root_y) / (3 + 1 + rows + 0.5))

    # The file and the two things in it.
    draw.text((root_x, root_y), "database.jpg", font=font, fill="#20303c")
    draw.text((root_x, root_y + line), "├── JPEG picture", font=font, fill="#17202a")
    duckdb_y = root_y + 2 * line
    draw.text((root_x, duckdb_y), "└── database.duckdb", font=font, fill="#17202a")

    # The schemas hang from database.duckdb. The stem starts under the middle
    # of the "d" and every column stem sits under the middle of its label's
    # first character, the same place the "├" and "└" strokes are drawn.
    col_a_x = root_x + 4 * char_width
    widest_a = max(
        draw.textlength(f"└── {name}", font=font)
        for _, tables in SCHEMA_TREE_COL_A
        for name in [*tables]
    )
    widest_b = max(
        draw.textlength(f"└── {name}", font=font)
        for _, tables in SCHEMA_TREE_COL_B
        for name in tables
    )
    # keep column B inside the panel's right margin
    col_b_x = min(col_a_x + widest_a + 26, x1 - 24 - widest_b)
    stem_a = col_a_x + char_width / 2
    stem_b = col_b_x + char_width / 2
    branch_y = duckdb_y + line + 2
    tree_top = branch_y + 12
    draw.line(
        (stem_a, duckdb_y + size + 2, stem_a, tree_top), fill=TREE_LINE_COLOR, width=3
    )
    draw.line((stem_a, branch_y, stem_b, branch_y), fill=TREE_LINE_COLOR, width=3)
    draw.line((stem_b, branch_y, stem_b, tree_top), fill=TREE_LINE_COLOR, width=3)

    _draw_tree(draw, font, SCHEMA_TREE_COL_A, origin=(col_a_x, tree_top), line=line)
    _draw_tree(draw, font, SCHEMA_TREE_COL_B, origin=(col_b_x, tree_top), line=line)
    _draw_caption(draw, fonts, box)


CAPTION_HEIGHT = 196
CAPTION_HEADLINE = "This picture is the database."
CAPTION_BODY = (
    "The file holds a photo first. A ZIP archive with the tables "
    "follows it. Picture viewers read the photo. DuckDB and ZIP tools "
    "read the archive."
)


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    box: tuple[int, int, int, int],
) -> None:
    x0, _y0, x1, y1 = box
    top = y1 - CAPTION_HEIGHT
    draw.line((x0 + 40, top, x1 - 40, top), fill="#d8ddd2", width=2)
    draw.text(
        (x0 + 40, top + 14), CAPTION_HEADLINE, font=fonts["subtitle"], fill="#20303c"
    )
    font = fonts["body_small"]
    max_width = (x1 - x0) - 80
    lines: list[str] = []
    line = ""
    for word in CAPTION_BODY.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    draw.multiline_text(
        (x0 + 40, top + 58),
        "\n".join(lines),
        font=font,
        fill="#17202a",
        spacing=6,
    )


def _draw_guide_panel(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, _, _ = box
    draw.text(
        (x0 + 40, y0 + 24),
        "One file, two ways in",
        font=fonts["heading"],
        fill="#20303c",
    )
    guide = (
        "Open as a picture: double-click database.jpg\n"
        "\n"
        "Open as a database:\n"
        "  -- run `duckdb` in this file's folder\n"
        "  INSTALL zipfs FROM community; LOAD zipfs;\n"
        "  INSTALL vss; LOAD vss;\n"
        "  SET zipfs_split = '!!';\n"
        "  SET hnsw_enable_experimental_persistence\n"
        "    = true;\n"
        "  COPY (SELECT content FROM read_blob(\n"
        "    'zip://database.jpg!!database.duckdb'))\n"
        "    TO 'database.duckdb' (FORMAT blob);\n"
        "  ATTACH 'database.duckdb' AS database\n"
        "    (READ_ONLY);\n"
        "  SELECT * FROM database.bbbc039.objects\n"
        "  LIMIT 10;"
    )
    draw.multiline_text(
        (x0 + 40, y0 + 90),
        guide,
        font=fonts["mono_small"],
        fill="#17202a",
        spacing=12,
    )


def _fonts() -> dict[str, ImageFont.ImageFont]:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    mono_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]

    def load(paths: list[str], size: int) -> ImageFont.ImageFont:
        for candidate in paths:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    return {
        "title": load(candidates, 66),
        "subtitle": load(candidates, 30),
        "heading": load(candidates, 38),
        "body_small": load(candidates, 27),
        "mono": load(mono_candidates + candidates, 29),
        "mono_schema": load(mono_candidates + candidates, 22),
        "mono_small": load(mono_candidates + candidates, 22),
    }


def _draw_tree(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    tree: list[tuple[str, list[str]]],
    origin: tuple[float, float] = (0, 0),
    line: float | None = None,
) -> None:
    """Render a real directory-tree-style listing (like the `tree` command):
    one top-level line per schema, its tables as branches underneath."""

    x, y = origin
    line_height = line or getattr(font, "size", 26) + 12
    group_gap = line_height // 2
    for i, (namespace, tables) in enumerate(tree):
        if i > 0:
            y += group_gap
        draw.text((x, y), namespace, font=font, fill="#3c7d89")
        y += line_height
        for j, table in enumerate(tables):
            branch = "└── " if j == len(tables) - 1 else "├── "
            draw.text((x, y), f"{branch}{table}", font=font, fill="#17202a")
            y += line_height


def _append_zip(paths: ArtifactPaths) -> None:
    with zipfile.ZipFile(
        paths.jpeg,
        mode="a",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.write(paths.duckdb, arcname="database.duckdb")
        archive.write(paths.readme, arcname="README.md")


def _write_browser_page(path: Path, counts: dict[str, int]) -> None:
    html = (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A picture is worth a thousand records</title>
  <link rel="icon" href="data:image/svg+xml,<svg
        xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22
        ><text y=%22.9em%22 font-size=%2290%22>%F0%9F%A6%86</text></svg>">
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f7f2;
      --muted: #b8c8d2;
      --line: #415363;
      --panel: #17232d;
      --accent: #79c2d0;
      --button: #e9f2ee;
      --button-ink: #17232d;
    }
    body {
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #111820;
      color: var(--ink);
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      max-width: 880px;
      margin: 0 auto;
    }
    .page-head {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px 20px;
      margin: 0 0 20px;
    }
    h1 {
      flex: 1 1 420px;
      margin: 0;
      font-size: clamp(32px, 6.4vw, 48px);
      line-height: 1.1;
      letter-spacing: -0.01em;
      text-wrap: balance;
    }
    .page-footer {
      margin: 36px 0 12px;
      padding-top: 20px;
      border-top: 1px solid var(--line);
      text-align: center;
    }
    .page-footer p {
      margin: 0 0 10px;
      font-size: 13px;
    }
    .page-footer .footer-title {
      font-size: 20px;
      font-weight: 700;
      color: var(--ink);
    }
    .footer-links {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px 22px;
    }
    .fig {
      display: block;
      width: 100%;
      height: auto;
      margin: 12px 0 18px;
      background: #0c1218;
      border: 1px solid var(--line);
    }
    .fig text {
      fill: var(--muted);
      font: 15px system-ui, sans-serif;
    }
    .fig .t-ink {
      fill: var(--ink);
      font-weight: 700;
    }
    .fig .t-dark {
      fill: #0c1218;
      font-weight: 700;
      font-size: 14px;
    }
    .fig .t-small {
      font-size: 13.5px;
    }
    .fig .t-tiny {
      font-size: 11px;
    }
    .fig .t-mid {
      text-anchor: middle;
    }
    .fig .box {
      fill: #131d26;
      stroke: #415363;
      stroke-width: 1.5;
    }
    .fig .line {
      stroke: #6b7f8e;
      stroke-width: 1.5;
    }
    .fig .mark-yes {
      fill: #79c2d0;
      font: 700 18px system-ui, sans-serif;
    }
    .fig .mark-no {
      fill: #e1096f;
      font: 700 18px system-ui, sans-serif;
    }
    .title-nowrap {
      white-space: nowrap;
    }
    .title-picture {
      color: __LOGO_BLUE__;
    }
    .title-records {
      color: __LOGO_PINK__;
    }
    .source-link {
      margin-top: 8px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 13px;
      text-decoration: none;
    }
    .source-link:hover {
      color: var(--ink);
      border-color: var(--accent);
    }
    .source-link svg {
      flex: none;
    }
    p {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.45;
    }
    a {
      color: var(--accent);
    }
    img.cover {
      width: 100%;
      height: auto;
      display: block;
      box-shadow: 0 18px 60px #0008;
      margin-bottom: 20px;
    }
    .panel {
      background: var(--panel);
      padding: 18px;
      margin-bottom: 20px;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 18px;
    }
    .text-section {
      margin: 30px 0;
      padding-top: 22px;
      border-top: 1px solid #2b3a47;
    }
    .text-section p:last-child {
      margin-bottom: 0;
    }
    .text-section ul:not(.citations) {
      margin: 0 0 16px;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.5;
    }
    .overlap-note ul {
      margin: 10px 0 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.5;
    }
    .diagram {
      margin: 14px 0 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .file-strip {
      display: grid;
      grid-template-columns: minmax(120px, 1.1fr) minmax(180px, 2fr);
      border: 1px solid var(--line);
      background: #0c1218;
    }
    .file-part {
      min-height: 72px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
      border-right: 1px solid var(--line);
    }
    .file-part:last-child {
      border-right: 0;
    }
    .file-part strong,
    .flow-step strong,
    .pipeline-step strong {
      color: var(--ink);
      font-size: 14px;
    }
    .file-part span,
    .flow-step span,
    .pipeline-step span {
      display: block;
    }
    .file-end {
      color: var(--accent);
    }
    .icon {
      font-size: 22px;
      line-height: 1.1;
    }
    /* inline emoji inside a label; the blocks above make every span a block */
    .emoji {
      display: inline !important;
    }
    .flow-diagram {
      display: grid;
      grid-template-columns: 1fr auto 1fr auto 1fr;
      gap: 10px;
      align-items: stretch;
    }
    .flow-step {
      min-height: 76px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #0c1218;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
    }
    .flow-arrow {
      align-self: center;
      color: var(--accent);
      font-size: 20px;
      line-height: 1;
    }
    .pipeline-diagram {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .pipeline-step {
      min-height: 86px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #0c1218;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
    }
    .pipeline-step::before {
      content: attr(data-step);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .dataset-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0 18px;
      margin: 8px 0 18px;
    }
    .dataset-stat {
      padding: 10px 0;
      border-top: 1px solid #2b3a47;
    }
    .dataset-stat strong {
      display: block;
      color: var(--ink);
      font-size: 22px;
      line-height: 1.1;
    }
    .dataset-stat span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      margin-top: 3px;
    }
    @media (max-width: 700px) {
      .file-strip,
      .flow-diagram,
      .pipeline-diagram {
        grid-template-columns: 1fr;
      }
      .dataset-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .file-part {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .file-part:last-child {
        border-bottom: 0;
      }
      .flow-arrow {
        transform: rotate(90deg);
        justify-self: center;
      }
    }
    .plot-intro {
      font-size: 13px;
      margin: 0 0 10px;
    }
    .cluster-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      margin-bottom: 8px;
    }
    .feature-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 16px;
    }
    .feature-tabs button[aria-pressed="true"] {
      background: var(--accent);
      color: #0c1218;
    }
    .cluster-grid .panel {
      margin-bottom: 0;
    }
    .maps-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .map-panel {
      padding: 10px;
      min-width: 0;
    }
    .map-panel h2 {
      margin: 0 0 4px;
      font-size: 16px;
    }
    .map-panel .plot-intro {
      min-height: 5.8em;
      margin: 0 0 8px;
      font-size: 12px;
      line-height: 1.4;
    }
    .map-panel .plot-toolbar {
      margin-bottom: 8px;
      gap: 6px;
    }
    .map-panel .plot-toolbar button {
      padding: 5px 8px;
      font-size: 12px;
    }
    .map-panel .plot-toolbar label {
      flex: 1 1 100%;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .map-panel .plot-toolbar input[type="range"] {
      flex: 1;
      min-width: 0;
    }
    .map-controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 16px;
      margin: 0 0 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .map-controls label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .plot-readout {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin: 8px 0 0;
      min-height: 11.5em;
      font-size: 12px;
      line-height: 1.4;
      color: var(--muted);
    }
    .plot-readout .readout-search {
      color: var(--ink);
    }
    @media (max-width: 820px) {
      .maps-grid {
        grid-template-columns: 1fr;
      }
      .map-panel .plot-intro {
        min-height: 0;
      }
    }
    .cluster-canvas {
      width: 100%;
      height: auto;
      max-width: 100%;
      display: block;
      background: #0c1218;
    }
    .plot-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      margin-bottom: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .plot-toolbar button {
      padding: 6px 12px;
      font-size: 13px;
    }
    .plot-toolbar label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .cluster-canvas {
      touch-action: none;
      cursor: grab;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      margin: 12px 0 20px;
      font-size: 13px;
      color: var(--muted);
    }
    .legend .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
    }
    .density-scale {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .density-ramp {
      display: inline-block;
      width: 90px;
      height: 10px;
      border-radius: 5px;
      background: linear-gradient(90deg, __DENSITY_GRADIENT__);
    }
    .legend .swatch.line {
      width: 14px;
      height: 2px;
      border-radius: 0;
      vertical-align: middle;
    }
    .tooltip {
      position: fixed;
      z-index: 20;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 6px;
      background: var(--panel);
      border: 1px solid var(--accent);
      pointer-events: none;
      font-size: 12px;
      color: var(--ink);
    }
    .tooltip[hidden] {
      display: none;
    }
    .tooltip img {
      width: 72px;
      height: 72px;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      display: block;
    }
    @media (max-width: 700px) {
      .cluster-grid {
        grid-template-columns: 1fr;
      }
    }
    .citations {
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.5;
    }
    .citations li {
      margin-bottom: 10px;
    }
    .search-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
    }
    input[type="number"] {
      min-height: 44px;
      padding: 0 12px;
      border: 1px solid var(--line);
      background: #0c1218;
      color: var(--ink);
      font: inherit;
      min-width: 140px;
      flex: 0 1 140px;
    }
    button {
      min-height: 44px;
      padding: 0 16px;
      border: 0;
      background: var(--button);
      color: var(--button-ink);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    button.secondary {
      background: transparent;
      border: 1px solid var(--line);
      color: var(--ink);
    }
    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }
    pre {
      margin: 0 0 16px;
      padding: 12px;
      overflow: auto;
      background: #0c1218;
      border: 1px solid var(--line);
      color: var(--muted);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(160px, 220px) 1fr;
      gap: 24px;
      align-items: start;
    }
    figure {
      margin: 0;
    }
    figure img {
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      display: block;
      image-rendering: pixelated;
    }
    figcaption {
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 14px;
    }
    .grid button.result {
      display: block;
      width: 100%;
      padding: 0;
      background: none;
      border: 1px solid var(--line);
      text-align: left;
      color: var(--ink);
      overflow: hidden;
    }
    .grid button.result img {
      width: 100%;
      height: auto;
      display: block;
      image-rendering: pixelated;
    }
    .grid button.result .meta {
      padding: 6px 8px;
      font-size: 12px;
      font-weight: 400;
      color: var(--muted);
      white-space: normal;
      overflow-wrap: break-word;
    }
    .grid button.result .meta span {
      display: block;
    }
    .grid button.result .meta .distance {
      color: var(--accent);
    }
    .grid button.result:hover {
      border-color: var(--accent);
    }
    @media (max-width: 700px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
    button.example {
      min-height: 32px;
      padding: 0 10px;
      font-size: 12px;
      font-weight: 400;
      background: transparent;
      border: 1px solid var(--line);
      color: var(--muted);
    }
    button.example:hover {
      border-color: var(--accent);
      color: var(--ink);
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    textarea {
      width: 100%;
      min-height: 140px;
      padding: 10px;
      box-sizing: border-box;
      background: #0c1218;
      color: var(--ink);
      border: 1px solid var(--line);
      font: 13px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
      resize: vertical;
      margin-bottom: 10px;
    }
    table {
      width: 100%;
      margin-top: 16px;
      border-collapse: collapse;
      font-size: 14px;
    }
    th,
    td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--accent);
      font-weight: 700;
    }
  </style>
</head>
<body>
  <main>
    <header class="page-head" id="top">
      <h1>
        A <span class="title-picture">picture</span> is worth
        <span class="title-nowrap">a thousand
          <span class="title-records">records</span></span>
      </h1>
      <a class="source-link" href="__REPO_URL__" target="_blank"
         rel="noopener">
        <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"
             aria-hidden="true">
          <path d="M8 0C3.58 0 0 3.58 0 8 c0 3.54 2.29 6.53 5.47 7.59.4.07.55
                   -.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69
                   -.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63
                   -.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28
                   -.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82
                   -2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18
                   1.32 -.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82
                   .44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07 -1.87
                   3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93 -.01
                   2.2 0 .21.15.46.55.38 A8.013 8.013 0 0016 8c0-4.42 -3.58-8-8-8
                   z"/>
        </svg>
        <span>Source on GitHub</span>
      </a>
    </header>
    <img class="cover" src="database.jpg"
         alt="database.jpg: the project logo, a BBBC039 microscope field,
         the database schema, and the SQL to open it">
    <p>
      A lot of biology starts with a microscope image. In
      <a href="https://doi.org/10.1038/nmeth.4397"
         target="_blank" rel="noopener">image-based profiling</a>,
      researchers find every cell or nucleus in an image, measure
      its size, shape, brightness, and texture, and compare those
      measurements across many samples. Increasingly they also pass each cell
      through a deep-learning model and keep the result, an
      <a href="https://en.wikipedia.org/wiki/Embedding_(machine_learning)"
         target="_blank" rel="noopener">embedding</a>: a
      list of numbers that summarizes how the cell looks.
    </p>
    <p>
      Along the way the data scatters. The images live in one place, the
      <a href="https://en.wikipedia.org/wiki/Image_segmentation"
         target="_blank" rel="noopener">segmentation</a>
      masks in another, and the measurement tables and
      embeddings somewhere else, often in formats that only the original
      pipeline can read. The picture that started the analysis ends up far
      from the numbers that describe it.
    </p>
    <p>
      This experiment asks whether one image file can carry both its data and
      the evidence needed to inspect a segmentation model's failures.
    </p>
    <p>
      This project tries the opposite. <code>database.jpg</code> is an
      ordinary JPEG: open it and you see the cover above, with the schema and
      the instructions to query it. Appended to the same file is a complete
      <a href="https://duckdb.org/"
         target="_blank" rel="noopener">DuckDB</a> database of __NUCLEUS_COUNT__
      hand-annotated nuclei from the public
      <a href="https://bbbc.broadinstitute.org/BBBC039"
         target="_blank" rel="noopener">BBBC039</a> benchmark.
      For every nucleus it holds an image crop, 271
      <a href="https://cellprofiler.org/"
         target="_blank" rel="noopener">CellProfiler</a>-style
      measurements, a learned embedding, and search
      indexes. This page downloads that one file and queries it in your
      browser. There is no database server, and nothing is uploaded.
    </p>
    <p>
      Scroll down to try it: run SQL against the file, compare three 3D maps
      of the nuclei
      (<a href="https://en.wikipedia.org/wiki/Principal_component_analysis"
         target="_blank" rel="noopener">PCA</a>,
      <a href="https://arxiv.org/abs/1802.03426"
         target="_blank" rel="noopener">UMAP</a>, and
      <a href="https://www.jmlr.org/papers/v9/vandermaaten08a.html"
         target="_blank" rel="noopener">t-SNE</a>), and
      search for look-alikes. The
      sections after that explain how one file can be both a picture and a
      database, where the data comes from, and what the comparison can and
      cannot tell you.
    </p>
    <div class="panel">
      <h2>Run your own SQL</h2>
      <p>
        This box runs SQL directly against the database inside
        <code>database.jpg</code>, using
        <a href="https://github.com/duckdb/duckdb-wasm"
           target="_blank" rel="noopener">DuckDB-Wasm</a>, DuckDB
        compiled to <a href="https://webassembly.org/"
           target="_blank" rel="noopener">WebAssembly</a>.
        Start from an example and edit it, or write your own. The maps and
        the search below read the same tables, so anything you see on this
        page you can also query here.
      </p>
      <div class="examples" id="sql-examples"></div>
      <textarea id="sql-editor" spellcheck="false"></textarea>
      <button id="run-sql" type="button" disabled>Run query</button>
      <pre id="sql-status">Waiting for database.jpg to load...</pre>
      <div id="sql-results" aria-live="polite"></div>
    </div>
    <h2>Compare the nuclei in feature space</h2>
    <p>
      Each point below is one hand-annotated nucleus, drawn three times. The
      three maps are three different ways to squeeze the same feature
      vectors into 3D: PCA, UMAP, and t-SNE. Nuclei that look alike should
      sit near each other, but "look alike" depends on how you describe a
      nucleus, and each map keeps different parts of the neighborhood
      structure. Switch between the three feature spaces to see how the
      representation reshapes all three maps at once.
    </p>
    <p>
      The points also carry a verdict from Cellpose, the segmentation model:
      __CELLPOSE_CAPTURE_PCT__% of these hand-annotated nuclei were captured
      by a Cellpose mask (IoU ≥ 0.5) on the same fields. IoU means
      <a href="https://en.wikipedia.org/wiki/Jaccard_index" target="_blank"
         rel="noopener">intersection over union</a>: the shared area divided
      by the total area covered by either mask. Tick
      <em>Highlight nuclei Cellpose missed</em> in the map controls to color
      the ones it missed, and see whether they cluster in any of the maps.
    </p>
    <div class="panel overlap-note">
      <h2>How to read the Cellpose overlap</h2>
      <p>
        The violet points are the __CELLPOSE_MISSED_COUNT__ manual nuclei
        (__CELLPOSE_MISSED_PCT__%) no Cellpose mask captured at IoU ≥ 0.5.
        They are overlaid on feature spaces computed from manual masks and DNA
        crops; the feature vectors did not include the Cellpose result. If
        violet points bunch up, that is a clue that missed nuclei share shape,
        intensity, or embedding neighborhoods. If they are scattered, the miss
        may come from field context, touching cells, or segmentation behavior
        these nucleus-level features do not capture.
      </p>
      <ul>
        <li><strong>cp_measure</strong> can expose simple measured traits:
          size, shape, brightness, texture, and radial intensity. It cannot see
          boundary ambiguity beyond the manual mask and intensity
          measurements.</li>
        <li><strong>MorphEm</strong> can group nuclei by learned visual
          appearance in the DNA crop, including patterns not named by
          hand-engineered measurements. It cannot tell you which measurement
          caused a group, and it may group visual similarity unrelated to
          Cellpose failure.</li>
        <li><strong>Fused</strong> asks where those two descriptions agree by
          combining reduced cp_measure and MorphEm vectors. It can make shared
          signals easier to find, but it cannot decide which side explains a
          miss without querying the source columns and crops.</li>
      </ul>
    </div>
    <div class="feature-tabs" id="feature-tabs" aria-label="Feature space"></div>
    <div class="map-controls">
      <label>
        <input id="auto-rotate" type="checkbox" checked>
        Slowly rotate all three
      </label>
      <label>
        <input id="highlight-missed" type="checkbox" checked>
        Highlight nuclei Cellpose missed
      </label>
      <button id="reset-all" class="secondary" type="button">
        Reset all views
      </button>
      <span class="plot-hint">
        Each map has its own view. Drag to rotate, scroll to zoom toward the
        cursor, Shift-drag to pan, and click a point to search. A map pauses
        its rotation while your pointer is over it.
      </span>
    </div>
    <div class="cluster-grid maps-grid">
      <figure class="panel map-panel" data-method="pca">
        <h2>PCA</h2>
        <p class="plot-intro">
          Linear. Keeps the directions of greatest variance, so global layout is
          faithful but neighborhoods can overlap.
        </p>
        <div class="plot-toolbar">
          <button class="secondary plot-zoom-selection" type="button">
            Zoom to selection
          </button>
          <button class="secondary plot-reset" type="button">Reset</button>
          <label>
            Zoom
            <input class="plot-zoom" type="range" min="0" max="100"
                   value="0" aria-label="PCA zoom">
          </label>
        </div>
        <canvas class="cluster-canvas" width="1000" height="1000"
                role="img" aria-label="Rotatable 3D PCA point cloud of
                BBBC039 feature vectors colored by neighbor density. Drag to
                rotate, scroll to zoom, click a point to search."></canvas>
        <p class="plot-readout" aria-live="polite">
          <span class="readout-search"></span>
          <span class="readout-all"></span>
        </p>
      </figure>
      <figure class="panel map-panel" data-method="umap">
        <h2>UMAP</h2>
        <p class="plot-intro">
          Non-linear. Rebuilds each nucleus's local neighborhood in 3D and keeps
          more of the global shape than t-SNE.
        </p>
        <div class="plot-toolbar">
          <button class="secondary plot-zoom-selection" type="button">
            Zoom to selection
          </button>
          <button class="secondary plot-reset" type="button">Reset</button>
          <label>
            Zoom
            <input class="plot-zoom" type="range" min="0" max="100"
                   value="0" aria-label="UMAP zoom">
          </label>
        </div>
        <canvas class="cluster-canvas" width="1000" height="1000"
                role="img" aria-label="Rotatable 3D UMAP point cloud of
                BBBC039 feature vectors colored by neighbor density. Drag to
                rotate, scroll to zoom, click a point to search."></canvas>
        <p class="plot-readout" aria-live="polite">
          <span class="readout-search"></span>
          <span class="readout-all"></span>
        </p>
      </figure>
      <figure class="panel map-panel" data-method="tsne">
        <h2>t-SNE</h2>
        <p class="plot-intro">
          Non-linear. Pulls near neighbors together tightly. Cluster
          sizes and gaps mean little.
        </p>
        <div class="plot-toolbar">
          <button class="secondary plot-zoom-selection" type="button">
            Zoom to selection
          </button>
          <button class="secondary plot-reset" type="button">Reset</button>
          <label>
            Zoom
            <input class="plot-zoom" type="range" min="0" max="100"
                   value="0" aria-label="t-SNE zoom">
          </label>
        </div>
        <canvas class="cluster-canvas" width="1000" height="1000"
                role="img" aria-label="Rotatable 3D t-SNE point cloud of
                BBBC039 feature vectors colored by neighbor density. Drag to
                rotate, scroll to zoom, click a point to search."></canvas>
        <p class="plot-readout" aria-live="polite">
          <span class="readout-search"></span>
          <span class="readout-all"></span>
        </p>
      </figure>
    </div>
    <div class="legend">
      <span>
        <span class="swatch" style="background:#e07a5f"></span>
        search seed
      </span>
      <span>
        <span class="swatch" style="background:#79c2d0"></span>
        12 nearest neighbors, found by the feature-space index (numbered by
        rank)
      </span>
      <span>
        <span class="swatch line" style="background:#f6f7f2"></span>
        6 nearest-neighbor edges
      </span>
      <span>
        <span class="swatch" style="background:__MISSED_COLOR__"></span>
        missed by Cellpose (tick <em>Highlight nuclei Cellpose missed</em>)
      </span>
      <span class="density-scale">
        sparse neighbors
        <span class="density-ramp" aria-hidden="true"></span>
        dense neighbors
      </span>
    </div>
    <p>
      Search for a nucleus by object number, or let the page pick one. The
      page shows its 12 most similar nuclei, ranked by distance in the
      current feature space. The same seed and neighbors are marked on all
      three maps, so you can compare how PCA, UMAP, and t-SNE treat one
      neighborhood: the index finds the neighbors in the full feature space,
      and each map shows how well it kept them together.
    </p>
    <div class="search-row">
      <input id="object-number" type="number" min="1" placeholder="Nucleus #"
             autocomplete="off">
      <button id="search" type="button" disabled>Find similar</button>
      <button id="random" class="secondary" type="button" disabled>
        Random nucleus
      </button>
    </div>
    <pre id="status">Loading database.jpg into DuckDB-Wasm...</pre>
    <p id="picker" hidden>
      Could not fetch <code>database.jpg</code> automatically (this happens
      when the page is opened from disk). Choose the file yourself. It stays
      in your browser and is never uploaded:
      <input id="picker-input" type="file" accept=".jpg,.jpeg,image/jpeg">
    </p>
    <div class="layout">
      <figure id="seed-figure" hidden>
        <img id="seed-image" alt="Search seed nucleus">
        <figcaption id="seed-caption"></figcaption>
      </figure>
      <div id="results" class="grid" aria-live="polite"></div>
    </div>
    <div id="hover-tooltip" class="tooltip" hidden>
      <img id="tooltip-image" alt="">
      <span id="tooltip-label"></span>
    </div>
    <section class="text-section">
      <h2>What is inside database.jpg</h2>
      <div class="diagram file-strip"
           aria-label="JPEG picture followed by ZIP archive">
        <div class="file-part">
          <span class="icon" aria-hidden="true">🖼️</span>
          <strong>JPEG picture</strong>
          <span>Opens in image viewers</span>
          <span class="file-end">
            <span class="emoji" aria-hidden="true">🏁</span> Ends at FF D9
          </span>
        </div>
        <div class="file-part">
          <span class="icon" aria-hidden="true">📦</span>
          <strong>ZIP archive</strong>
          <span>
            <span class="emoji" aria-hidden="true">🦆</span>
            <code>database.duckdb</code>
          </span>
          <span>
            <span class="emoji" aria-hidden="true">📄</span>
            <code>README.md</code>
          </span>
        </div>
      </div>
      <p>
        <code>database.jpg</code> is a
        <a href="https://en.wikipedia.org/wiki/Polyglot_(computing)"
           target="_blank" rel="noopener">polyglot</a>:
        one sequence of bytes that is
        valid in two file formats. The first part is a JPEG picture. The
        second part is a ZIP archive that holds two files.
      </p>
      <ul>
        <li><span class="emoji" aria-hidden="true">🦆</span>
          <code>database.duckdb</code>: the database. DuckDB (a program
          that keeps tables in one file) reads it.</li>
        <li><span class="emoji" aria-hidden="true">📄</span>
          <code>README.md</code>: the instructions for the database.</li>
      </ul>
      <h3>How can one file be both?</h3>
      __DIAGRAM_POLYGLOT__
      <p>
        A <a href="https://en.wikipedia.org/wiki/JPEG"
           target="_blank" rel="noopener">JPEG</a> ends with a
        <a href="https://en.wikipedia.org/wiki/JPEG#Syntax_and_structure"
           target="_blank" rel="noopener">two-byte marker</a>,
        <code>FF D9</code>. The marker
        means that the picture ends here. A picture viewer stops at this
        marker and ignores every byte after it.
      </p>
      <p>
        A <a href="https://en.wikipedia.org/wiki/ZIP_(file_format)"
           target="_blank" rel="noopener">ZIP archive</a>
        keeps its table of contents at the end of the file.
        A ZIP tool reads the end first. The table of contents tells the tool
        where each file starts, so the tool never needs the bytes at the
        front.
      </p>
      <p>
        Because the two formats read from opposite ends, each tool sees only
        its own part. A picture viewer sees a picture, and a ZIP tool sees an
        archive. The archive stores the database without compression, so the
        database inside it is a byte-for-byte copy of
        <code>database.duckdb</code>, and DuckDB can read it straight out of
        the JPEG.
      </p>
      <p>
        One practical warning: keep the original file. Many apps and
        websites re-encode pictures when you send or upload them, and a
        re-encoded JPEG keeps the photo but loses everything after
        <code>FF D9</code>.
      </p>
    </section>
    <section class="text-section">
      <h2>Why this shape?</h2>
      <div class="diagram flow-diagram"
           aria-label="Image leads to analysis data in one file">
        <div class="flow-step">
          <span class="icon" aria-hidden="true">🖼️</span>
          <strong>Image first</strong>
          <span>The source you can see</span>
        </div>
        <div class="flow-arrow" aria-hidden="true">→</div>
        <div class="flow-step">
          <span class="icon" aria-hidden="true">📊</span>
          <strong>Analysis data</strong>
          <span>Cells, crops, tables, embeddings</span>
        </div>
        <div class="flow-arrow" aria-hidden="true">→</div>
        <div class="flow-step">
          <span class="icon" aria-hidden="true">🗃️</span>
          <strong>One file</strong>
          <span>Look first, query next</span>
        </div>
      </div>
      <p>
        Scientific data formats usually treat images as payload: arrays to be
        stored and decoded, not something you look at first. We wanted the
        opposite order. The first thing anyone sees when they open this file
        is what the data is about: the source image, the schema, and how to
        query it. The analysis, down to individual nuclei, sits behind that
        picture in the same file.
      </p>
      <p>
        Two recent projects shaped the design.
        <a href="https://doi.org/10.1145/3749163" target="_blank"
           rel="noopener">F3</a> is a data file format that ships the data,
        its metadata, and WebAssembly decoders together, so a file carries the
        means to read it.
        <a href="https://arxiv.org/abs/2608.07632" target="_blank"
           rel="noopener">JUMP-lite</a> packages image-based profiling data
        in a compact, reproducible form for benchmarking cell
        representations. This project takes a simpler, deliberately
        low-tech path: a standard JPEG, a standard ZIP, and a standard DuckDB
        file, read by tools that already exist.
      </p>
      <p>
        The trade-offs are real. A single file is easy to share and cite. But
        you must download all of it before the page can query it, and it
        grows with every table you add.
      </p>
      <h3>Why keep it under 100 MB?</h3>
      __DIAGRAM_SIZE__
      <p>
        <a href="https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github"
           target="_blank" rel="noopener">GitHub blocks any file larger than
        100 MiB</a> (about 105 MB). This project keeps its code, its data,
        and this page in one GitHub repository. So the data file must fit
        under that limit. This file is __JPEG_MB__.
      </p>
      <p>
        To fit, this file leaves out the original images and masks. It stores
        measurements in single precision. It keeps search indexes for only two
        of the three feature spaces.
      </p>
      <p>
        Other options exist. Git LFS (large file storage) keeps big files
        outside the repository, and you can host a file somewhere else. Both
        add another system to set up. We chose one small file instead.
      </p>
    </section>
    <section class="text-section">
      <h2>How this all works</h2>
      <div class="diagram pipeline-diagram"
           aria-label="Pipeline from source images to browser queries">
        <div class="pipeline-step" data-step="1">
          <span class="icon" aria-hidden="true">🔬</span>
          <strong>BBBC039 images</strong>
          <span>200 DNA fields from U2OS cells</span>
        </div>
        <div class="pipeline-step" data-step="2">
          <span class="icon" aria-hidden="true">✏️</span>
          <strong>Manual masks</strong>
          <span>Each mask marks nucleus objects.</span>
        </div>
        <div class="pipeline-step" data-step="3">
          <span class="icon" aria-hidden="true">📏🧠</span>
          <strong>Measure and embed</strong>
          <span>Measurements and MorphEm make feature vectors.</span>
        </div>
        <div class="pipeline-step" data-step="4">
          <span class="icon" aria-hidden="true">🔎</span>
          <strong>Query and compare</strong>
          <span>DuckDB compares isolated and fused spaces.</span>
        </div>
      </div>
      <h3>Dataset at a glance</h3>
      <div class="dataset-stats" aria-label="Dataset counts">
        <div class="dataset-stat">
          <strong>__IMAGE_COUNT__</strong>
          <span><span class="emoji" aria-hidden="true">🖼️</span> Images</span>
        </div>
        <div class="dataset-stat">
          <strong>__NUCLEUS_COUNT__</strong>
          <span><span class="emoji" aria-hidden="true">🧫</span> Nuclei</span>
        </div>
        <div class="dataset-stat">
          <strong>__TRAINING_IMAGES__</strong>
          <span><span class="emoji" aria-hidden="true">🏋️</span> Training images</span>
        </div>
        <div class="dataset-stat">
          <strong>__VALIDATION_IMAGES__</strong>
          <span>
            <span class="emoji" aria-hidden="true">✅</span> Validation images
          </span>
        </div>
        <div class="dataset-stat">
          <strong>__TEST_IMAGES__</strong>
          <span><span class="emoji" aria-hidden="true">🧪</span> Test images</span>
        </div>
        <div class="dataset-stat">
          <strong>__FEATURE_SPACES__</strong>
          <span><span class="emoji" aria-hidden="true">🗺️</span> Feature spaces</span>
        </div>
      </div>
      <p>
        So the title undersells it a bit: this picture is worth more than a
        thousand records. Count the crops, feature vectors, and neighbor
        edges, and it carries hundreds of thousands of database rows.
      </p>
      <p>
        The data comes from
        <a href="https://bbbc.broadinstitute.org/BBBC039" target="_blank"
           rel="noopener">BBBC039</a>, a public benchmark from the
        <a href="https://bbbc.broadinstitute.org/"
           target="_blank" rel="noopener">Broad Bioimage Benchmark Collection</a>.
        It has 200 fluorescence images of human
        <a href="https://www.cellosaurus.org/CVCL_0042"
           target="_blank" rel="noopener">U2OS</a>
        cells, stained so that DNA glows, and a hand-drawn mask outlining
        every nucleus. Using human annotations instead of an automatic
        segmentation means the nuclei themselves are not in question: any
        difference you see comes from how they are described.
      </p>
      <p>
        That description is the core question of the field, and the page
        compares three answers. The first is
        <a href="__CP_MEASURE_URL__" target="_blank"
           rel="noopener">cp_measure</a>, a Python library that computes
        <a href="https://cellprofiler.org/"
           target="_blank" rel="noopener">CellProfiler</a>'s measurements: 271
        values for size, shape, DNA
        intensity, texture, granularity, and radial distribution. We ran it
        on the manual masks, then reduced 263 of the values to 16 numbers
        with
        <a href="https://en.wikipedia.org/wiki/Principal_component_analysis"
           target="_blank" rel="noopener">PCA</a>. The full measurements are in
        <code>bbbc039.objects</code>. The second is the pretrained
        <a href="https://huggingface.co/CaicedoLab/MorphEm" target="_blank"
           rel="noopener">MorphEm</a> model, a
        <a href="https://arxiv.org/abs/2010.11929"
           target="_blank" rel="noopener">vision transformer</a> trained
        on microscopy images, to embed each nucleus crop. Instead of measuring
        named traits, the model reads the DNA crop and turns it into numbers
        learned from image examples. It captures patterns nobody named in
        advance, but its 384 numbers are hard to interpret one by one. The
        third, the fused vector, joins both after
        <a href="https://en.wikipedia.org/wiki/Dimensionality_reduction"
           target="_blank" rel="noopener">dimension reduction</a>,
        to test whether the two views complement each other.
      </p>
      <p>
        Finding similar nuclei is a
        <a href="https://en.wikipedia.org/wiki/Nearest_neighbor_search"
           target="_blank" rel="noopener">nearest-neighbor search</a>.
        Similarity is plain
        <a href="https://en.wikipedia.org/wiki/Euclidean_distance"
           target="_blank" rel="noopener">Euclidean distance</a>
        between vectors, computed in SQL with DuckDB's
        <a href="https://duckdb.org/docs/current/sql/functions/array.html"
           target="_blank" rel="noopener"><code>array_distance</code></a>. The maps are
        built once per feature space, when the file is built, and are only for
        looking. The search results, not the maps, are the ground truth: a
        neighbor that looks far away on a map is still a true neighbor.
      </p>
      <p>
        In the cp_measure and Fused spaces, the search does not compare the
        chosen nucleus against every other one. The database holds an
        <a href="https://duckdb.org/docs/current/core_extensions/vss"
           target="_blank" rel="noopener">HNSW</a> index
        (<a href="https://arxiv.org/abs/1603.09320"
           target="_blank" rel="noopener">Malkov &amp; Yashunin, 2016</a>)
        for each of them. DuckDB's <code>vss</code>
        extension builds and reads these indexes. When you search, DuckDB asks
        the index for the nearest nuclei, and the page shows them. At build
        time we compared each index with an exact scan on about 200 nuclei.
        The index returned at least __HNSW_RECALL__ of the exact 12 nearest
        neighbors. MorphEm has no stored index, because its 384-number
        vectors make the index about 33 MB. MorphEm searches compare the
        chosen nucleus against all __NUCLEUS_COUNT__ vectors instead, which
        is still fast at this size. The message under the search box tells
        you which method answered each search. Each map draws lines from the
        selected nucleus to its 6 nearest neighbors.
      </p>
      <h3>Why use an index?</h3>
      __DIAGRAM_INDEX__
      <p>
        An index is a shortcut that makes search faster. Without one, the
        page compares your nucleus with every other nucleus. This is an exact
        scan (a full comparison). It is always right, but it gets slow when a
        collection has millions of cells.
      </p>
      <p>
        <a href="https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world"
           target="_blank" rel="noopener">HNSW</a> (Hierarchical Navigable
        Small World) is one kind of index. It links each nucleus to a few
        close neighbors, in layers. The top layer has few nuclei and long
        links, and the bottom layer has every nucleus and short links. A
        search starts at the top. It moves down one layer at a time, always
        toward your nucleus, like going from country to city to street. It
        visits only a small part of the data.
      </p>
      <p>
        The shortcut has two costs. The answer is approximate, so the index
        can miss a true neighbor (the check above found few misses). The
        index also takes space in the file. A MorphEm index adds about 33 MB,
        which takes the file past the 100 MB limit. So MorphEm has no index,
        and its searches use an exact scan.
      </p>
      <p>
        With 19,565 nuclei, an exact scan is already fast. Here the index is a
        demonstration that pays off on larger collections. It also travels
        inside the same file as the data.
      </p>
      <h3>Three ways to draw the same vectors</h3>
      __DIAGRAM_MAPS__
      <p>
        A nucleus is a list of 16, 48, or 384 numbers, and nobody can look at
        384 dimensions. Dimension reduction squeezes those vectors into three
        so you can rotate them. It always loses something. The three maps
        lose different things, so putting them side by side shows what each
        one can and cannot be trusted with.
      </p>
      <ul>
        <li>
          <a href="https://en.wikipedia.org/wiki/Principal_component_analysis"
             target="_blank" rel="noopener">PCA</a>
          (<a href="https://doi.org/10.1098/rsta.2015.0202"
             target="_blank" rel="noopener">Jolliffe &amp; Cadima, 2016</a>)
          is linear. It rotates the data so the
          first axis captures the most variance, the second the next most,
          and so on, then keeps the first three. The layout is faithful to
          global distances, but three axes rarely hold enough of the
          variance, so different neighborhoods land on top of each other.
        </li>
        <li>
          <a href="https://umap-learn.readthedocs.io/" target="_blank"
             rel="noopener">UMAP</a>
          (<a href="https://arxiv.org/abs/1802.03426"
             target="_blank" rel="noopener">McInnes, Healy &amp; Melville, 2018</a>)
          is non-linear. It builds a graph of each nucleus's nearest neighbors
          and then lays the graph out in 3D. It keeps neighborhoods well and
          usually more of the overall shape than t-SNE, but distances between
          clusters are only roughly meaningful. The interactive essay
          <a href="https://pair-code.github.io/understanding-umap/"
             target="_blank" rel="noopener">Understanding UMAP</a>
          shows how its settings change the picture.
        </li>
        <li>
          <a href="https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding"
             target="_blank" rel="noopener">t-SNE</a>
          (<a href="https://www.jmlr.org/papers/v9/vandermaaten08a.html"
             target="_blank" rel="noopener">van der Maaten &amp; Hinton, 2008</a>),
          here computed with
          <a href="https://opentsne.readthedocs.io/en/stable/"
             target="_blank" rel="noopener">openTSNE</a>
          (<a href="https://doi.org/10.18637/jss.v109.i03"
             target="_blank" rel="noopener">Policar, Strazar &amp; Zupan, 2024</a>),
          is also non-linear and puts the most weight on keeping near
          neighbors together. It is very good at that, and it often makes
          tight, well-separated blobs. But the size of a blob and the gaps
          between blobs carry little meaning, as the essay
          <a href="https://distill.pub/2016/misread-tsne/"
             target="_blank" rel="noopener">How to Use t-SNE Effectively</a>
          shows with many small examples.
        </li>
      </ul>
      <p>
        Each map's caption reports two numbers, measured over all nuclei
        against exact nearest neighbors in the feature space. The first is
        the share of a nucleus's 6 true neighbors that are also among its 15
        nearest points on the map. The second is how many times closer a true
        neighbor sits than a random nucleus. When you run a search, each map
        also reports how much closer the 12 retrieved neighbors sit to the
        seed than chance. Expect PCA to score lowest on these and t-SNE
        highest, because t-SNE is built to optimize exactly this. That does
        not make t-SNE the best picture: it can exaggerate clusters that the
        vectors do not really separate.
      </p>
      <h3>What is neighbor density?</h3>
      __DIAGRAM_DENSITY__
      <p>
        Neighbor density is the average distance from a nucleus to its 6
        nearest neighbors in the feature space. A short distance means that
        many similar nuclei sit close by, so the neighborhood is dense. A
        long distance means the nucleus is unusual, so its neighborhood is
        sparse. The page ranks all nuclei by this number. It colors the
        ranks from blue (sparsest) to yellow (densest).
      </p>
      <p>
        None of the three maps keeps density faithfully. A crowded patch on a
        map is not always a dense group of nuclei, and t-SNE in particular
        evens out density on purpose. So the color adds information that the
        position of a point does not show.
      </p>
      <p>
        HNSW finds these neighbors. At build time, HNSW indexes with the same
        settings as the stored ones find the 6 nearest neighbors of every
        nucleus in each feature space. The build saves them in
        <code>features.knn_graph</code>. For MorphEm the build uses a
        temporary index that it does not keep. When the page loads, it
        averages the 6 distances for each nucleus. When you search, the
        stored cp_measure and Fused indexes find the neighbors of the nucleus
        you pick.
      </p>
    </section>
    <section class="text-section">
      <h2>What does fusion show?</h2>
      __DIAGRAM_FUSION__
      <p>
        The three feature-space buttons use the same nuclei, the same search
        code, and the same distance. Only the vector changes, so any
        difference in who counts as a neighbor comes from the representation
        alone.
        Here, fusion is simple concatenation: we put the 16 cp_measure
        numbers next to 32 PCA-reduced MorphEm numbers, making one 48-number
        vector for each nucleus.
      </p>
      <p>
        That makes the page a way to build intuition about a live debate in
        the field. Engineered measurements like cp_measure's are transparent
        and long established: you can say a nucleus was grouped with others
        because it is large and bright. Learned embeddings like MorphEm's
        often separate subtle phenotypes better but are harder to explain.
        Fusing them asks whether you can have some of both.
      </p>
      <h3>What this is not</h3>
      __DIAGRAM_NOT__
      <p>
        This is an interactive experiment, not a benchmark result. We did not
        score the feature spaces against a biological ground truth, and
        BBBC039 was designed for segmentation, not for comparing treatments.
        Some other limits to keep in mind:
      </p>
      <ul>
        <li>The database stores 8-bit JPEG crops for display. It does not
          store the original 16-bit images or the masks, which you can
          download from BBBC039.</li>
        <li>Measurements are stored as
          <a href="https://en.wikipedia.org/wiki/Single-precision_floating-point_format"
             target="_blank" rel="noopener">32-bit floats</a>, and each image's
          intensities were scaled to its own brightest pixel before
          measuring, so intensity values compare best within an image.</li>
        <li>The HNSW index is approximate, and DuckDB still marks storing
          it in a file as
          <a href="https://duckdb.org/docs/current/core_extensions/vss"
             target="_blank" rel="noopener">experimental</a>.</li>
        <li>UMAP and t-SNE layouts depend on their settings (for t-SNE, mainly
          the <a href="https://distill.pub/2016/misread-tsne/"
             target="_blank" rel="noopener">perplexity</a>) and on the random
          seed, and PCA keeps only three axes. Treat the shape of any one map
          as a sketch, not a measurement.</li>
      </ul>
    </section>
    <section class="text-section">
      <h2>What the picture carries</h2>
      __DIAGRAM_CARRIES__
      <p>
        We started with a simple question: what if the picture carried the
        data that came from it? In this experiment, a single JPEG under
        100 MB holds __NUCLEUS_COUNT__ nuclei. It stores their image crops,
        271 measurements each, learned embeddings, search indexes, and
        neighbor lists. It opens as a picture in any image viewer and as a
        database in DuckDB. It also powers the interactive page you are
        reading, with no server behind it.
      </p>
      <p>
        The three maps show one thing clearly. A 3D map is a summary, and
        different summaries keep different things.
        We measured each map against exact neighbors in the feature space.
        In all three feature spaces, t-SNE kept the most of each nucleus's
        true neighborhood and PCA the least. Even so, no map replaces the
        answer it draws. The search runs in the full feature space, so you
        can check any map against the result it tries to show.
      </p>
      <p>
        This page is an experiment. It shows what is possible when one
        picture file carries its own data. It is not a recommendation or a
        standard.
      </p>
      <p>
        Everything on this page is open. <a href="database.jpg"
           target="_blank" rel="noopener">Download database.jpg</a>,
        open it with the DuckDB recipe printed on the cover, and ask your own
        questions. The code that built it is on
        <a href="__REPO_URL__"
           target="_blank" rel="noopener">GitHub</a>, and every figure you see
        here comes from tables you can query with the SQL box above.
      </p>
    </section>
    <section class="text-section">
      <h2>References</h2>
      <p>
        Here is where the data, the models, and the methods on this page
        came from.
      </p>
      <h3>Software</h3>
      <p>
        These packages do the technical work in this build.
      </p>
      <ul>
        <li><a href="https://bbbc.broadinstitute.org/BBBC039" target="_blank"
          rel="noopener">BBBC039</a>: source images and manual masks.</li>
        <li><a href="https://scikit-image.org/" target="_blank"
          rel="noopener">scikit-image</a>: decoding the masks into labeled
          nuclei.</li>
        <li><a href="__CP_MEASURE_URL__" target="_blank"
          rel="noopener">cp_measure</a>: the CellProfiler
          measurements.</li>
        <li><a href="https://huggingface.co/CaicedoLab/MorphEm"
          target="_blank" rel="noopener">CaicedoLab/MorphEm</a>: nucleus
          embeddings.</li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"
          target="_blank" rel="noopener">scikit-learn</a>: the PCA
          coordinates and the exact neighbor checks.</li>
        <li><a href="https://umap-learn.readthedocs.io/" target="_blank"
          rel="noopener">umap-learn</a>: UMAP coordinates.</li>
        <li><a href="https://opentsne.readthedocs.io/en/stable/"
          target="_blank" rel="noopener">openTSNE</a>: t-SNE coordinates.</li>
        <li><a href="https://duckdb.org/docs/current/core_extensions/vss"
          target="_blank" rel="noopener">DuckDB vss</a>: the HNSW indexes
          and the nearest-neighbor search.</li>
        <li><a href="https://duckdb.org/" target="_blank"
          rel="noopener">DuckDB</a> and
          <a href="https://github.com/duckdb/duckdb-wasm" target="_blank"
          rel="noopener">DuckDB-Wasm</a>: SQL in the database and browser.</li>
      </ul>
      <h3>Papers and data</h3>
      <ul class="citations">
        <li>
          Zeng, X., Meng, R., Prammer, M., McKinney, W., Patel, J.M.,
          Pavlo, A., et al. (2025).
          <a href="https://doi.org/10.1145/3749163" target="_blank"
             rel="noopener">F3: The Open-Source Data File Format for the
          Future</a>. Proceedings of the ACM on Management of Data, 3(4),
          1 to 27. Inspiration for a file that carries both data and the
          means to read it.
        </li>
        <li>
          Caicedo, J.C., Cooper, S., Heigwer, F., et al. (2017).
          <a href="https://doi.org/10.1038/nmeth.4397" target="_blank"
             rel="noopener">Data-analysis strategies for image-based cell
          profiling</a>. Nature Methods, 14, 849&ndash;863. An overview of
          the field this page works in.
        </li>
        <li>
          Ljosa, V., Sokolnicki, K.L., &amp; Carpenter, A.E. (2012).
          <a href="https://doi.org/10.1038/nmeth.2083" target="_blank"
             rel="noopener">Annotated high-throughput microscopy image sets
          for validation</a>. Nature Methods, 9, 637. Source publication for
          the Broad Bioimage Benchmark Collection.
        </li>
        <li>
          Caicedo, J.C., Goodman, A., Karhohs, K.W., et al. (2019).
          <a href="https://doi.org/10.1002/cyto.a.23863" target="_blank"
             rel="noopener">Nucleus segmentation across imaging experiments:
          the 2018 Data Science Bowl</a>. Cytometry Part A, 95(9),
          952 to 965. Recommended source for BBBC039v1.
        </li>
        <li>
          <a href="https://bbbc.broadinstitute.org/BBBC039" target="_blank"
             rel="noopener">BBBC039</a>. Nuclei of U2OS cells in a chemical
          screen. Source of the 200 DNA images and manual masks.
        </li>
        <li>
          van der Walt, S., Schonberger, J.L., Nunez-Iglesias, J., et al.
          (2014). <a href="https://doi.org/10.7717/peerj.453"
             target="_blank" rel="noopener">scikit-image: image processing
          in Python</a>. PeerJ, 2, e453. Used to decode the masks into
          labeled nuclei.
        </li>
        <li>
          Munoz, A.F., Treis, T., Kalinin, A.A., Dasgupta, S., Theis, F.,
          Carpenter, A.E., &amp; Singh, S. (2025).
          <a href="__CP_MEASURE_PAPER_URL__" target="_blank"
             rel="noopener">cp_measure: API-first feature extraction for
          image-based profiling workflows</a>. arXiv:2507.01163. The library
          (version __CP_MEASURE_VERSION__) that computed the 271 CellProfiler
          measurements for each nucleus.
        </li>
        <li>
          Carpenter, A.E., Jones, T.R., Lamprecht, M.R., et al. (2006).
          <a href="https://doi.org/10.1186/gb-2006-7-10-r100"
             target="_blank" rel="noopener">CellProfiler: image analysis
          software for identifying and quantifying cell phenotypes</a>.
          Genome Biology, 7, R100. Original source for the CellProfiler
          measurement style that cp_measure implements.
        </li>
        <li>
          Stringer, C., Wang, T., Michaelos, M., &amp; Pachitariu, M. (2021).
          <a href="https://doi.org/10.1038/s41592-020-01018-x"
             target="_blank" rel="noopener">Cellpose: a generalist algorithm
          for cellular segmentation</a>. Nature Methods, 18, 100&ndash;106.
          Background for the segmentation model family used in the comparison.
        </li>
        <li>
          Pachitariu, M., Rariden, M., &amp; Stringer, C. (2025).
          <a href="https://doi.org/10.1101/2025.04.28.651001"
             target="_blank" rel="noopener">Cellpose-SAM: superhuman
          generalization for cellular segmentation</a>. bioRxiv. Reference
          for the Cellpose-SAM model family behind <code>cpsam_v2</code>.
        </li>
        <li>
          <a href="https://en.wikipedia.org/wiki/Jaccard_index"
             target="_blank" rel="noopener">Jaccard index</a>. Reference for
          intersection over union (IoU), the overlap metric used to mark
          Cellpose captures and misses.
        </li>
        <li>
          Chen, Z., Pham, C., Wang, S., Doron, M., Moshkov, N., Plummer,
          B.A., &amp; Caicedo, J.C. (2023).
          <a href="https://arxiv.org/abs/2310.19224" target="_blank"
             rel="noopener">CHAMMI: A benchmark for channel-adaptive models
          in microscopy imaging</a>. NeurIPS Datasets and Benchmarks Track.
          The dataset MorphEm trained on.
        </li>
        <li>
          <a href="https://huggingface.co/CaicedoLab/MorphEm" target="_blank"
             rel="noopener">CaicedoLab/MorphEm</a>. The pretrained model
          used to compute each nucleus embedding.
        </li>
        <li>
          Munoz, A.F., Haslum, J.F., Shen, R., Carpenter, A.E., &amp;
          Singh, S. (2026).
          <a href="https://arxiv.org/abs/2608.07632" target="_blank"
             rel="noopener">JUMP-lite: Compact, reproducible benchmarking of
          cell representations</a>. arXiv:2608.07632. Benchmarks
          cp_measure and MorphEm using lossy JPEG XL compression.
        </li>
        <li>
          Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).
          <a href="https://arxiv.org/abs/2010.11929" target="_blank"
             rel="noopener">An Image is Worth 16x16 Words: Transformers for
          Image Recognition at Scale</a>. ICLR 2021. The vision transformer
          architecture MorphEm builds on.
        </li>
        <li>
          Jolliffe, I.T., &amp; Cadima, J. (2016).
          <a href="https://doi.org/10.1098/rsta.2015.0202" target="_blank"
             rel="noopener">Principal component analysis: a review and recent
          developments</a>. Philosophical Transactions of the Royal Society A,
          374(2065), 20150202.
        </li>
        <li>
          van der Maaten, L., &amp; Hinton, G. (2008).
          <a href="https://www.jmlr.org/papers/v9/vandermaaten08a.html"
             target="_blank" rel="noopener">Visualizing data using
          t-SNE</a>. Journal of Machine Learning Research, 9, 2579&ndash;2605.
        </li>
        <li>
          van der Maaten, L. (2014).
          <a href="https://jmlr.org/papers/v15/vandermaaten14a.html"
             target="_blank" rel="noopener">Accelerating t-SNE using
          tree-based algorithms</a>. Journal of Machine Learning Research, 15,
          3221&ndash;3245. The Barnes-Hut method used for the 3D t-SNE maps.
        </li>
        <li>
          Policar, P.G., Strazar, M., &amp; Zupan, B. (2024).
          <a href="https://doi.org/10.18637/jss.v109.i03" target="_blank"
             rel="noopener">openTSNE: a modular Python library for t-SNE
          dimensionality reduction and embedding</a>. Journal of Statistical
          Software, 109(3).
        </li>
        <li>
          McInnes, L., Healy, J., &amp; Melville, J. (2018).
          <a href="https://arxiv.org/abs/1802.03426" target="_blank"
             rel="noopener">UMAP: Uniform Manifold Approximation and
          Projection for Dimension Reduction</a>. arXiv:1802.03426.
        </li>
        <li>
          Malkov, Y.A., &amp; Yashunin, D.A. (2016).
          <a href="https://arxiv.org/abs/1603.09320" target="_blank"
             rel="noopener">Efficient and robust approximate nearest
          neighbor search using Hierarchical Navigable Small World
          graphs</a>. arXiv:1603.09320.
        </li>
        <li>
          Raasveldt, M., &amp; Muhleisen, H. (2019).
          <a href="https://doi.org/10.1145/3299869.3320212" target="_blank"
             rel="noopener">DuckDB: an Embeddable Analytical Database</a>.
          SIGMOD 2019.
        </li>
      </ul>
  </section>
    <footer class="page-footer">
      <p class="footer-title">
        A <span class="title-picture">picture</span> is worth
        <span class="title-nowrap">a thousand
          <span class="title-records">records</span></span>.
      </p>
      <p>
        This page is one picture file, one small database, and the code that
        built them. The code uses the BSD 3-Clause license. The BBBC039 images
        and masks are CC0.
      </p>
      <p class="footer-links">
        <a href="__REPO_URL__"
           target="_blank" rel="noopener">Source on GitHub</a>
        <a href="database.jpg">Download database.jpg</a>
        <a href="#top">Back to top</a>
      </p>
    </footer>
  </main>
  <script type="importmap">
    {
      "imports": {
        "/npm/": "https://cdn.jsdelivr.net/npm/"
      }
    }
  </script>
  <script type="module">
    import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.33.1-dev57.0/+esm";
    import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.2/+esm";

    const statusEl = document.querySelector("#status");
    const featureTabs = document.querySelector("#feature-tabs");
    const searchButton = document.querySelector("#search");
    const randomButton = document.querySelector("#random");
    const objectNumberInput = document.querySelector("#object-number");
    const seedFigure = document.querySelector("#seed-figure");
    const seedImage = document.querySelector("#seed-image");
    const seedCaption = document.querySelector("#seed-caption");
    const resultsEl = document.querySelector("#results");
    const tooltip = document.querySelector("#hover-tooltip");
    const tooltipImage = document.querySelector("#tooltip-image");
    const tooltipLabel = document.querySelector("#tooltip-label");
    // the tooltip is position:fixed and only moves on mousemove, so without
    // this it stays put (over whatever content scrolled underneath it)
    // instead of following the plot box as the page scrolls.
    window.addEventListener(
      "scroll",
      () => {
        tooltip.hidden = true;
      },
      { passive: true, capture: true },
    );

    const setStatus = (message) => {
      statusEl.textContent = message;
    };

    const jpegToDataUrl = (bytes) => {
      // pixel_data_jpeg is already a real, plain JPEG, no decoding
      // needed, just base64 it straight into a data URL.
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(
          null,
          bytes.subarray(i, i + chunkSize),
        );
      }
      return `data:image/jpeg;base64,${btoa(binary)}`;
    };

    let connection = null;
    let hnswAvailable = false;
    let activeFeatureSpace = "__DEFAULT_FEATURE_SPACE__";
    let dataset = null;
    // ObjectNumber -> 1 when Cellpose missed that nucleus; powers the hover
    // tooltip and the "Highlight nuclei Cellpose missed" toggle.
    let missedById = new Map();
    let quality = new Map();
    let cropImageCache = new Map();
    let hoverRequestId = 0;
    const DEFAULT_OBJECT_NUMBER = "269";
    const FEATURE_SPACES = {
      cp_measure: {
        label: "cp_measure",
        table: "database.features.cp_measure",
        vectorColumn: "measurement_vector",
        dim: 16,
        indexed: true,
      },
      morphem: {
        label: "MorphEm",
        table: "database.features.morphem",
        vectorColumn: "embedding",
        dim: 384,
        indexed: false,
      },
      fused: {
        label: "Fused",
        table: "database.features.fused",
        vectorColumn: "fused_embedding",
        dim: 48,
        indexed: true,
      },
    };

    const fetchJpegBytes = async () => {
      setStatus("Fetching database.jpg...");
      const response = await fetch("database.jpg", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`database.jpg returned ${response.status}`);
      }
      return response.arrayBuffer();
    };

    const openDatabase = async (jpegBytes) => {
      setStatus("Extracting database.duckdb from JPEG ZIP payload...");
      const zip = await JSZip.loadAsync(jpegBytes);
      const databaseMember = zip.file("database.duckdb");
      if (!databaseMember) {
        throw new Error("database.duckdb was not found in the JPEG ZIP payload");
      }
      const databaseBytes = await databaseMember.async("uint8array");

      setStatus("Starting DuckDB-Wasm...");
      const bundles = duckdb.getJsDelivrBundles();
      const bundle = await duckdb.selectBundle(bundles);
      const worker = await duckdb.createWorker(bundle.mainWorker);
      const database = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
      await database.instantiate(bundle.mainModule, bundle.pthreadWorker);
      await database.registerFileBuffer("database.duckdb", databaseBytes);

      const conn = await database.connect();
      // The database holds HNSW indexes for cp_measure and fused. DuckDB's
      // vss extension must be loaded to use them. Without it the data still
      // opens and the same SQL runs as an exact scan (as it always does for
      // MorphEm, which has no stored index).
      setStatus("Loading the DuckDB vss extension...");
      try {
        await conn.query("INSTALL vss");
        await conn.query("LOAD vss");
        await conn.query("SET hnsw_enable_experimental_persistence = true");
        hnswAvailable = true;
      } catch (error) {
        console.warn("vss extension unavailable; searching without HNSW", error);
      }
      await conn.query("ATTACH 'database.duckdb' AS database (READ_ONLY)");
      return conn;
    };

    const cellValue = (value) => (typeof value === "bigint" ? value.toString() : value);

    const normalizeObjectNumber = (value) => {
      const objectNumber = Number(value);
      if (!Number.isInteger(objectNumber) || objectNumber < 1) {
        return null;
      }
      return String(objectNumber);
    };

    // generic scalar formatter for the free-form SQL console below, where a
    // result column can be any real type, including BLOBs and the
    // FLOAT[1152] embedding array, unlike the fixed queries above.
    const sqlCellValue = (value) => {
      if (value === null || value === undefined) {
        return "";
      }
      if (typeof value === "bigint") {
        return value.toString();
      }
      if (value instanceof Uint8Array) {
        return `<${value.byteLength} bytes>`;
      }
      if (ArrayBuffer.isView(value) || Array.isArray(value)) {
        return `[${value.length} values]`;
      }
      if (typeof value === "object" && typeof value.toArray === "function") {
        return `[${value.toArray().length} values]`;
      }
      return value;
    };

    const rowsOf = (table) => table.toArray().map((row) => row.toJSON());

    const DEFAULT_VIEW = { yaw: 0.65, pitch: 0.35, zoom: 1, tx: 0, ty: 0, tz: 0 };
    const MAX_ZOOM = 25;
    const OUTLIER_PULL = 0.35;
    const MAP_METHODS = ["pca", "umap", "tsne"];
    // The search result, drawn on all three maps.
    let selection = { seed: null, neighbors: [], hnsw: [] };
    const plots = [];

    const densityRanks = (distances) => {
      // Rank the nuclei by the average distance to their nearest neighbors,
      // so the color scale is spread evenly. A min-max scale would put almost
      // every point at one end, because a few outliers stretch the range.
      const ranked = distances
        .map((distance, i) => ({ i, distance }))
        .filter((entry) => Number.isFinite(entry.distance))
        .sort((a, b) => a.distance - b.distance);
      const density = new Float32Array(distances.length);
      const last = Math.max(1, ranked.length - 1);
      ranked.forEach((entry, rank) => {
        density[entry.i] = 1 - rank / last;
      });
      return density;
    };

    const loadClusterData = async () => {
      const feature = FEATURE_SPACES[activeFeatureSpace];
      // Keep the startup query small. Crop images are fetched on demand for
      // hover cards and nearest-neighbor results.
      const pointTable = await connection.query(`
        SELECT
            f.ObjectNumber,
            f.pca_x, f.pca_y, f.pca_z,
            f.umap_x, f.umap_y, f.umap_z,
            f.tsne_x, f.tsne_y, f.tsne_z,
            d.avg_neighbor_distance,
            o.cellpose_captured
        FROM ${feature.table} f
        LEFT JOIN (
            SELECT ObjectNumber, AVG(distance) AS avg_neighbor_distance
            FROM database.features.knn_graph
            WHERE feature_space = '${activeFeatureSpace}'
            GROUP BY ObjectNumber
        ) d USING (ObjectNumber)
        LEFT JOIN (
            SELECT ObjectNumber, cellpose_captured
            FROM database.bbbc039.objects
        ) o USING (ObjectNumber)
        ORDER BY f.ObjectNumber
      `);
      const rows = rowsOf(pointTable);
      const count = rows.length;
      const coords = Object.fromEntries(
        MAP_METHODS.map((method) => [method, new Float32Array(count * 3)]),
      );
      const ids = new Array(count);
      const distances = new Array(count);
      const missed = new Uint8Array(count);
      rows.forEach((row, i) => {
        ids[i] = String(cellValue(row.ObjectNumber));
        distances[i] = row.avg_neighbor_distance === null
          ? null
          : Number(row.avg_neighbor_distance);
        // Missed = the Cellpose masks did not capture this manual nucleus.
        missed[i] = row.cellpose_captured ? 0 : 1;
        for (const method of MAP_METHODS) {
          coords[method][i * 3] = Number(row[`${method}_x`]);
          coords[method][i * 3 + 1] = Number(row[`${method}_y`]);
          coords[method][i * 3 + 2] = Number(row[`${method}_z`]);
        }
      });
      dataset = {
        ids,
        density: densityRanks(distances),
        coords,
        missed,
      };
      missedById = new Map(ids.map((id, i) => [id, missed[i]]));
      const qualityTable = await connection.query(`
        SELECT method, neighbor_recall, distance_ratio, explained_variance
        FROM database.features.embedding_quality
        WHERE feature_space = '${activeFeatureSpace}'
      `);
      quality = new Map(
        rowsOf(qualityTable).map((row) => [row.method, {
          recall: Number(row.neighbor_recall),
          ratio: Number(row.distance_ratio),
          // only PCA has an explained variance; the rest are NULL
          variance: row.explained_variance === null
            ? NaN
            : Number(row.explained_variance),
        }]),
      );
      for (const plot of plots) {
        plot.setData(dataset);
      }
    };

    const rememberCropImage = (objectNumber, dataUrl) => {
      cropImageCache.set(objectNumber, dataUrl);
      if (cropImageCache.size > 64) {
        const oldestKey = cropImageCache.keys().next().value;
        cropImageCache.delete(oldestKey);
      }
    };

    const fetchCropImageDataUrl = async (objectNumber) => {
      const cached = cropImageCache.get(objectNumber);
      if (typeof cached === "string") {
        return cached;
      }
      if (cached) {
        return cached;
      }
      const request = connection.query(`
        SELECT pixel_data_jpeg
        FROM database.images.object_crops
        WHERE object_type = 'nucleus' AND channel = 'DNA'
            AND ObjectNumber = ${objectNumber}
        LIMIT 1
      `).then((table) => {
        const rows = rowsOf(table);
        if (rows.length === 0) {
          throw new Error(`No crop found for nucleus ${objectNumber}.`);
        }
        const dataUrl = jpegToDataUrl(rows[0].pixel_data_jpeg);
        rememberCropImage(objectNumber, dataUrl);
        return dataUrl;
      }).catch((error) => {
        cropImageCache.delete(objectNumber);
        throw error;
      });
      cropImageCache.set(objectNumber, request);
      return request;
    };

    const mixColor = (from, to, amount) => {
      const value = Math.max(0, Math.min(1, amount));
      return from.map((channel, index) => (
        Math.round(channel + (to[index] - channel) * value)
      ));
    };

    const rgb = (channels) => `rgb(${channels.join(", ")})`;

    const DENSITY_STOPS = __DENSITY_STOPS__;
    const densityColor = (density) => {
      const scaled = Math.max(0, Math.min(1, density)) * (DENSITY_STOPS.length - 1);
      const from = Math.min(DENSITY_STOPS.length - 2, Math.floor(scaled));
      return rgb(mixColor(
        DENSITY_STOPS[from],
        DENSITY_STOPS[from + 1],
        scaled - from,
      ));
    };

    // Perspective camera looking at the unit sphere from CAMERA_DISTANCE.
    const CAMERA_DISTANCE = 4;
    const DEPTH_BINS = 6;
    const DENSITY_BUCKETS = 8;
    const autoRotateInput = document.querySelector("#auto-rotate");
    const highlightMissedInput = document.querySelector("#highlight-missed");
    // Bright violet, used only for nuclei Cellpose missed.
    const MISSED_COLOR = "__MISSED_COLOR__";

    const showTooltip = (objectNumber, event) => {
      const requestId = hoverRequestId + 1;
      hoverRequestId = requestId;
      tooltipImage.removeAttribute("src");
      tooltipLabel.textContent = missedById.get(String(objectNumber))
        ? `nucleus ${objectNumber} — missed by Cellpose`
        : `nucleus ${objectNumber}`;
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
      tooltip.hidden = false;
      fetchCropImageDataUrl(objectNumber).then((dataUrl) => {
        if (requestId === hoverRequestId) {
          tooltipImage.src = dataUrl;
        }
      }).catch(() => {
        if (requestId === hoverRequestId) {
          tooltip.hidden = true;
        }
      });
    };

    const hideTooltip = () => {
      hoverRequestId += 1;
      tooltip.hidden = true;
    };

    // One rotatable, zoomable 3D scatter plot. Each map gets its own view, so
    // the three plots turn and zoom independently, but all three draw the same
    // search result.
    const createPlot = (panel) => {
      const method = panel.dataset.method;
      const canvas = panel.querySelector("canvas");
      const ctx = canvas.getContext("2d");
      const zoomSlider = panel.querySelector(".plot-zoom");
      const readoutSearch = panel.querySelector(".readout-search");
      const readoutAll = panel.querySelector(".readout-all");
      const view = { ...DEFAULT_VIEW };
      // The view a map opens with, and returns to on Reset. It is centered
      // on the points once the data arrives (see centerDefaultView).
      let defaultView = { ...DEFAULT_VIEW };
      let cloud = {
        ids: [],
        index: new Map(),
        x: null,
        y: null,
        z: null,
        density: null,
        missed: null,
        extent: 1,
        minZoom: 1,
        typicalGap: 1,
      };
      let projected = { sx: null, sy: null, depth: null, scale: null };
      let frameRequest = 0;
      let viewAnimation = 0;
      let pointerOver = false;
      let visible = true;
      let cssWidth = canvas.getBoundingClientRect().width || 400;
      // Sizes below are CSS pixels; the canvas has more pixels than that.
      const px = () => canvas.width / cssWidth;

      new ResizeObserver((entries) => {
        cssWidth = entries[0].contentRect.width || cssWidth;
        requestDraw();
      }).observe(canvas);
      new IntersectionObserver((entries) => {
        visible = entries[0].isIntersecting;
      }).observe(canvas);

      const setData = (data) => {
        const source = data.coords[method];
        const count = data.ids.length;
        // Center the cloud and scale it so 95% of the points fall inside the
        // unit sphere. The few far outliers would otherwise shrink everything,
        // so they are pulled toward the center, keeping their order.
        let cx = 0;
        let cy = 0;
        let cz = 0;
        for (let i = 0; i < count; i += 1) {
          cx += source[i * 3];
          cy += source[i * 3 + 1];
          cz += source[i * 3 + 2];
        }
        cx /= count;
        cy /= count;
        cz /= count;
        const radii = new Float64Array(count);
        for (let i = 0; i < count; i += 1) {
          radii[i] = Math.hypot(
            source[i * 3] - cx,
            source[i * 3 + 1] - cy,
            source[i * 3 + 2] - cz,
          );
        }
        const sorted = Float64Array.from(radii).sort();
        const r95 = sorted[Math.floor(0.95 * (count - 1))] || 1e-9;
        const pulled = (radius) => {
          const scaled = radius / r95;
          return scaled <= 1 ? scaled : 1 + (scaled - 1) * OUTLIER_PULL;
        };
        const extent = Math.max(1, pulled(sorted[count - 1]));
        const place = (value, center, radius) => (
          radius > 0 ? ((value - center) / radius) * pulled(radius) : 0
        );
        const x = new Float32Array(count);
        const y = new Float32Array(count);
        const z = new Float32Array(count);
        for (let i = 0; i < count; i += 1) {
          x[i] = place(source[i * 3], cx, radii[i]);
          y[i] = place(source[i * 3 + 1], cy, radii[i]);
          z[i] = place(source[i * 3 + 2], cz, radii[i]);
        }
        // The distance between two random nuclei on this map, so a search can
        // report how much closer its neighbors sit than chance.
        const gaps = [];
        let seed = 12345;
        const next = () => {
          seed = (seed * 1664525 + 1013904223) % 4294967296;
          return seed / 4294967296;
        };
        for (let k = 0; k < 4000; k += 1) {
          const a = Math.floor(next() * count);
          const b = Math.floor(next() * count);
          gaps.push(Math.hypot(x[a] - x[b], y[a] - y[b], z[a] - z[b]));
        }
        gaps.sort((a, b) => a - b);
        cloud = {
          ids: data.ids,
          index: new Map(data.ids.map((id, i) => [id, i])),
          x,
          y,
          z,
          density: data.density,
          missed: data.missed,
          extent,
          minZoom: Math.min(1, 1 / (extent * 1.05)),
          typicalGap: gaps[gaps.length >> 1] || 1,
        };
        projected = {
          sx: new Float32Array(count),
          sy: new Float32Array(count),
          depth: new Float32Array(count),
          scale: new Float32Array(count),
        };
        centerDefaultView();
        syncZoomSlider();
        describeQuality();
        setSelection(selection);
      };

      // Shift the default view up or down so the middle 99% of the points
      // sit centered on the canvas. An elongated cloud (PCA especially)
      // would otherwise run off the top or bottom edge.
      const centerDefaultView = () => {
        Object.assign(view, DEFAULT_VIEW);
        project();
        const sorted = Float32Array.from(projected.sy).sort();
        const last = sorted.length - 1;
        const low = sorted[Math.floor(0.005 * last)];
        const high = sorted[Math.floor(0.995 * last)];
        const { unit, half } = cameraFrame();
        const down = half - (low + high) / 2;
        const [wx, wy, wz] = screenToCloud(0, down / unit);
        defaultView = { ...DEFAULT_VIEW, tx: wx, ty: wy, tz: wz };
        Object.assign(view, defaultView);
      };

      const describeQuality = () => {
        const row = quality.get(method);
        if (!row) {
          readoutAll.textContent = "";
          return;
        }
        const variance = Number.isFinite(row.variance)
          ? ` The 3 axes explain ${(row.variance * 100).toFixed(0)}% of the variance.`
          : "";
        readoutAll.textContent =
          `All nuclei: ${(row.recall * 100).toFixed(0)}% of true neighbors stay `
          + `among the 15 nearest points on this map. A neighbor sits `
          + `${(1 / row.ratio).toFixed(0)}x closer than a random nucleus.${variance}`;
      };

      const describeSearch = () => {
        const seedIndex = cloud.index.get(selection.seed);
        const indexes = selection.neighbors
          .map((id) => cloud.index.get(id))
          .filter((i) => i !== undefined);
        if (seedIndex === undefined || !indexes.length) {
          readoutSearch.textContent = "Search for a nucleus to compare.";
          return;
        }
        const gaps = indexes
          .map((i) => Math.hypot(
            cloud.x[i] - cloud.x[seedIndex],
            cloud.y[i] - cloud.y[seedIndex],
            cloud.z[i] - cloud.z[seedIndex],
          ))
          .sort((a, b) => a - b);
        const median = gaps[gaps.length >> 1];
        const times = cloud.typicalGap / Math.max(median, 1e-9);
        readoutSearch.textContent =
          `This search: the median neighbor sits ${times.toFixed(1)}x closer `
          + "to the seed than a random nucleus does.";
      };

      // ----- camera and projection -----
      const cameraFrame = () => {
        const half = canvas.width / 2;
        // At zoom 1 the unit sphere (95% of the points) reaches the canvas
        // edge. Zoom out below 1 to bring the pulled-in outliers into view.
        const baseUnit = half * 0.98;
        return {
          half,
          baseUnit,
          unit: baseUnit * view.zoom,
          cosYaw: Math.cos(view.yaw),
          sinYaw: Math.sin(view.yaw),
          cosPitch: Math.cos(view.pitch),
          sinPitch: Math.sin(view.pitch),
        };
      };

      const project = () => {
        // Turn the cloud about the view center (tx, ty, tz), then apply
        // perspective and zoom.
        const frame = cameraFrame();
        const { half, unit, cosYaw, sinYaw, cosPitch, sinPitch } = frame;
        const { x, y, z } = cloud;
        const { sx, sy, depth, scale } = projected;
        for (let i = 0; i < x.length; i += 1) {
          const dx = x[i] - view.tx;
          const dy = y[i] - view.ty;
          const dz = z[i] - view.tz;
          const x1 = dx * cosYaw + dz * sinYaw;
          const z1 = -dx * sinYaw + dz * cosYaw;
          const y2 = dy * cosPitch - z1 * sinPitch;
          const z2 = dy * sinPitch + z1 * cosPitch;
          const f = CAMERA_DISTANCE / (CAMERA_DISTANCE - z2);
          sx[i] = half + x1 * f * unit;
          sy[i] = half - y2 * f * unit;
          depth[i] = z2;
          scale[i] = f;
        }
        return frame;
      };

      const selectedIndexes = () => {
        const seen = new Set();
        const indexes = [];
        for (const id of [...selection.neighbors, ...selection.hnsw]) {
          const i = cloud.index.get(id);
          if (i !== undefined && !seen.has(i)) {
            seen.add(i);
            indexes.push(i);
          }
        }
        return indexes;
      };

      const drawBackgroundDots = () => {
        // 19,565 overlapping dots are unreadable. Draw them small, fade the
        // far ones, and dim everything when a nucleus is selected.
        const size = px();
        const hasSelection = selection.seed !== null;
        const { sx, sy, depth } = projected;
        const edge = canvas.width + 10 * size;
        const bins = Array.from(
          { length: DEPTH_BINS * DENSITY_BUCKETS },
          () => [],
        );
        // When the toggle is on, missed nuclei skip the density ramp and get
        // the pink accent instead; they are drawn last within each depth
        // slice so they stay visible on top.
        const highlightMissed = highlightMissedInput.checked && cloud.missed;
        const missedBins = Array.from({ length: DEPTH_BINS }, () => []);
        for (let i = 0; i < sx.length; i += 1) {
          if (sx[i] < -10 * size || sx[i] > edge
              || sy[i] < -10 * size || sy[i] > edge) {
            continue;
          }
          const near = Math.min(
            DEPTH_BINS - 1,
            Math.max(
              0,
              Math.floor(((depth[i] / cloud.extent + 1) / 2) * DEPTH_BINS),
            ),
          );
          if (highlightMissed && cloud.missed[i]) {
            missedBins[near].push(sx[i], sy[i]);
            continue;
          }
          const bucket = Math.min(
            DENSITY_BUCKETS - 1,
            Math.floor(cloud.density[i] * DENSITY_BUCKETS),
          );
          bins[near * DENSITY_BUCKETS + bucket].push(sx[i], sy[i]);
        }
        ctx.save();
        for (let near = 0; near < DEPTH_BINS; near += 1) {
          const t = near / (DEPTH_BINS - 1);
          const radius = (1 + 1.1 * t) * size * Math.min(3, Math.sqrt(view.zoom));
          ctx.globalAlpha = (0.5 + 0.5 * t) * (hasSelection ? 0.7 : 1);
          for (let bucket = 0; bucket < DENSITY_BUCKETS; bucket += 1) {
            const coords = bins[near * DENSITY_BUCKETS + bucket];
            if (!coords.length) {
              continue;
            }
            ctx.fillStyle = densityColor((bucket + 0.5) / DENSITY_BUCKETS);
            ctx.beginPath();
            for (let i = 0; i < coords.length; i += 2) {
              ctx.moveTo(coords[i] + radius, coords[i + 1]);
              ctx.arc(coords[i], coords[i + 1], radius, 0, Math.PI * 2);
            }
            ctx.fill();
          }
          const missedCoords = missedBins[near];
          if (missedCoords.length) {
            ctx.fillStyle = MISSED_COLOR;
            ctx.beginPath();
            for (let i = 0; i < missedCoords.length; i += 2) {
              ctx.moveTo(missedCoords[i] + radius, missedCoords[i + 1]);
              ctx.arc(missedCoords[i], missedCoords[i + 1], radius, 0, Math.PI * 2);
            }
            ctx.fill();
          }
        }
        ctx.restore();
      };

      const drawSelection = () => {
        const seedIndex = cloud.index.get(selection.seed);
        if (seedIndex === undefined) {
          return;
        }
        const size = px();
        const { sx, sy, depth, scale } = projected;
        const grow = Math.min(1.8, Math.sqrt(view.zoom));
        // Lines from the seed to its 6 nearest neighbors.
        ctx.save();
        ctx.lineWidth = 1.5 * size;
        ctx.strokeStyle = "rgba(246, 247, 242, 0.6)";
        for (const id of selection.hnsw) {
          const i = cloud.index.get(id);
          if (i === undefined) {
            continue;
          }
          ctx.beginPath();
          ctx.moveTo(sx[seedIndex], sy[seedIndex]);
          ctx.lineTo(sx[i], sy[i]);
          ctx.stroke();
        }
        ctx.restore();
        // Neighbors far to near, then the seed on top.
        const neighbors = selection.neighbors
          .map((id, rank) => ({ i: cloud.index.get(id), rank }))
          .filter((entry) => entry.i !== undefined)
          .sort((a, b) => depth[a.i] - depth[b.i]);
        for (const { i, rank } of neighbors) {
          const radius = 6 * size * Math.min(1.35, Math.max(0.75, scale[i])) * grow;
          ctx.beginPath();
          ctx.arc(sx[i], sy[i], radius, 0, Math.PI * 2);
          ctx.fillStyle = "#79c2d0";
          ctx.fill();
          ctx.lineWidth = 2 * size;
          ctx.strokeStyle = "#0c1218";
          ctx.stroke();
          ctx.fillStyle = "#0c1218";
          ctx.font = `bold ${8 * size}px system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(rank + 1), sx[i], sy[i] + 0.5 * size);
        }
        ctx.beginPath();
        ctx.arc(
          sx[seedIndex],
          sy[seedIndex],
          8 * size * Math.min(1.35, Math.max(0.75, scale[seedIndex])) * grow,
          0,
          Math.PI * 2,
        );
        ctx.fillStyle = "#e07a5f";
        ctx.fill();
        ctx.lineWidth = 2.5 * size;
        ctx.strokeStyle = "#f6f7f2";
        ctx.stroke();
      };

      const draw = () => {
        if (!cloud.x) {
          return;
        }
        project();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawBackgroundDots();
        drawSelection();
      };

      const requestDraw = () => {
        if (!frameRequest) {
          frameRequest = requestAnimationFrame(() => {
            frameRequest = 0;
            draw();
          });
        }
      };

      const setSelection = (next) => {
        // `selection` is shared; each plot only redraws and re-describes.
        if (cloud.x) {
          describeSearch();
          draw();
        }
      };

      // ----- zoom, pan, and reset -----
      // The slider is logarithmic, from the zoom that shows every point up to
      // MAX_ZOOM.
      const zoomFromSlider = (value) => (
        cloud.minZoom * (MAX_ZOOM / cloud.minZoom) ** (value / 100)
      );

      const syncZoomSlider = () => {
        zoomSlider.value = String(
          (100 * Math.log(view.zoom / cloud.minZoom))
            / Math.log(MAX_ZOOM / cloud.minZoom),
        );
      };

      const setZoom = (zoom) => {
        view.zoom = Math.max(cloud.minZoom, Math.min(MAX_ZOOM, zoom));
        syncZoomSlider();
        requestDraw();
      };

      // Turn an offset in the screen plane (right, up) into cloud coordinates
      // with the inverse of the current rotation.
      const screenToCloud = (right, up) => {
        const { cosYaw, sinYaw, cosPitch, sinPitch } = cameraFrame();
        const y = up * cosPitch;
        const z1 = -up * sinPitch;
        return [
          right * cosYaw - z1 * sinYaw,
          y,
          right * sinYaw + z1 * cosYaw,
        ];
      };

      // Move the view center in the screen plane (Shift + drag).
      const panView = (start, dx, dy, canvasScale) => {
        const { unit } = cameraFrame();
        const [wx, wy, wz] = screenToCloud(
          (-dx * canvasScale) / unit,
          (dy * canvasScale) / unit,
        );
        view.tx = start.tx + wx;
        view.ty = start.ty + wy;
        view.tz = start.tz + wz;
      };

      // Zoom toward the point under the cursor, so it stays where it is.
      const zoomAt = (newZoom, x, y) => {
        const zoom = Math.max(cloud.minZoom, Math.min(MAX_ZOOM, newZoom));
        const ratio = zoom / view.zoom;
        const { half, unit } = cameraFrame();
        const right = (x - half) / unit;
        const up = -(y - half) / unit;
        const [wx, wy, wz] = screenToCloud(
          right * (1 - 1 / ratio),
          up * (1 - 1 / ratio),
        );
        view.tx += wx;
        view.ty += wy;
        view.tz += wz;
        view.zoom = zoom;
        syncZoomSlider();
        requestDraw();
      };

      const animateViewTo = (target) => {
        cancelAnimationFrame(viewAnimation);
        const from = { ...view };
        const start = performance.now();
        const step = (now) => {
          const t = Math.min(1, (now - start) / 350);
          const eased = 1 - (1 - t) ** 3;
          view.zoom = Math.exp(
            Math.log(from.zoom)
              + (Math.log(target.zoom) - Math.log(from.zoom)) * eased,
          );
          view.tx = from.tx + (target.tx - from.tx) * eased;
          view.ty = from.ty + (target.ty - from.ty) * eased;
          view.tz = from.tz + (target.tz - from.tz) * eased;
          syncZoomSlider();
          draw();
          if (t < 1) {
            viewAnimation = requestAnimationFrame(step);
          }
        };
        viewAnimation = requestAnimationFrame(step);
      };

      // Center the view on the seed and its neighbors and zoom to fit them.
      // This runs only when you press the button, never on its own.
      const zoomToSelection = () => {
        const indexes = selectedIndexes();
        const seedIndex = cloud.index.get(selection.seed);
        if (seedIndex !== undefined) {
          indexes.push(seedIndex);
        }
        if (!indexes.length) {
          return;
        }
        let cx = 0;
        let cy = 0;
        let cz = 0;
        for (const i of indexes) {
          cx += cloud.x[i];
          cy += cloud.y[i];
          cz += cloud.z[i];
        }
        cx /= indexes.length;
        cy /= indexes.length;
        cz /= indexes.length;
        let radius = 0.05;
        for (const i of indexes) {
          radius = Math.max(
            radius,
            Math.hypot(cloud.x[i] - cx, cloud.y[i] - cy, cloud.z[i] - cz),
          );
        }
        const { baseUnit } = cameraFrame();
        const zoom = Math.max(
          cloud.minZoom,
          Math.min(MAX_ZOOM, (0.36 * canvas.width) / (radius * baseUnit)),
        );
        animateViewTo({ zoom, tx: cx, ty: cy, tz: cz });
      };

      const resetView = () => {
        cancelAnimationFrame(viewAnimation);
        Object.assign(view, defaultView);
        syncZoomSlider();
        draw();
      };

      // ----- picking and pointer input -----
      const canvasCoords = (event) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
          x: (event.clientX - rect.left) * scaleX,
          y: (event.clientY - rect.top) * scaleY,
          scale: (scaleX + scaleY) / 2,
        };
      };

      const pickPoint = (x, y) => {
        // Prefer the seed and its neighbors. Otherwise take the dot under the
        // cursor that is nearest the camera, so a front dot wins over a hidden
        // one.
        const { sx, sy, depth } = projected;
        const radiusSq = (10 * px()) ** 2;
        let best = -1;
        let bestDepth = -Infinity;
        const seedIndex = cloud.index.get(selection.seed);
        const priority = selectedIndexes();
        if (seedIndex !== undefined) {
          priority.push(seedIndex);
        }
        for (const i of priority) {
          const distSq = (sx[i] - x) ** 2 + (sy[i] - y) ** 2;
          if (distSq <= radiusSq && depth[i] > bestDepth) {
            best = i;
            bestDepth = depth[i];
          }
        }
        if (best < 0) {
          for (let i = 0; i < sx.length; i += 1) {
            const distSq = (sx[i] - x) ** 2 + (sy[i] - y) ** 2;
            if (distSq <= radiusSq && depth[i] > bestDepth) {
              best = i;
              bestDepth = depth[i];
            }
          }
        }
        return best < 0 ? null : { objectNumber: cloud.ids[best], index: best };
      };

      let drag = null;
      let dragged = false;
      canvas.addEventListener("pointerdown", (event) => {
        canvas.setPointerCapture(event.pointerId);
        drag = {
          x: event.clientX,
          y: event.clientY,
          yaw: view.yaw,
          pitch: view.pitch,
          tx: view.tx,
          ty: view.ty,
          tz: view.tz,
          pan: event.shiftKey,
        };
        dragged = false;
      });
      canvas.addEventListener("pointerup", (event) => {
        canvas.releasePointerCapture(event.pointerId);
        drag = null;
        if (dragged) {
          canvas.style.cursor = "grab";
          return;
        }
        if (!cloud.x) {
          return;
        }
        const { x, y } = canvasCoords(event);
        const picked = pickPoint(x, y);
        if (picked) {
          runSearch(picked.objectNumber);
        }
      });
      canvas.addEventListener("pointermove", (event) => {
        if (!cloud.x) {
          return;
        }
        if (drag) {
          const dx = event.clientX - drag.x;
          const dy = event.clientY - drag.y;
          if (dragged || Math.hypot(dx, dy) > 4) {
            dragged = true;
            canvas.style.cursor = "grabbing";
            hideTooltip();
            if (drag.pan) {
              panView(drag, dx, dy, canvasCoords(event).scale);
            } else {
              view.yaw = drag.yaw + dx * 0.008;
              view.pitch = Math.max(
                -Math.PI / 2,
                Math.min(Math.PI / 2, drag.pitch + dy * 0.008),
              );
            }
            requestDraw();
          }
          return;
        }
        const { x, y } = canvasCoords(event);
        const nearest = pickPoint(x, y);
        if (!nearest) {
          hideTooltip();
          canvas.style.cursor = "grab";
          return;
        }
        canvas.style.cursor = "pointer";
        showTooltip(nearest.objectNumber, event);
      });
      canvas.addEventListener("pointerenter", () => {
        pointerOver = true;
      });
      canvas.addEventListener("pointerleave", () => {
        pointerOver = false;
        hideTooltip();
      });
      // The wheel zooms toward the cursor. At the zoom limits it hands the
      // scroll back to the page, so a map never traps the page. A trackpad
      // pinch arrives as a Ctrl + wheel event with small steps.
      canvas.addEventListener("wheel", (event) => {
        const atMin = view.zoom <= cloud.minZoom * 1.001 && event.deltaY > 0;
        const atMax = view.zoom >= MAX_ZOOM && event.deltaY < 0;
        if (!cloud.x || atMin || atMax) {
          return;
        }
        event.preventDefault();
        cancelAnimationFrame(viewAnimation);
        const lines = event.deltaMode === 1 ? 16 : 1;
        const step = event.ctrlKey ? 0.01 : 0.002;
        const { x, y } = canvasCoords(event);
        zoomAt(view.zoom * Math.exp(-event.deltaY * lines * step), x, y);
      }, { passive: false });

      panel.querySelector(".plot-zoom-selection")
        .addEventListener("click", zoomToSelection);
      panel.querySelector(".plot-reset").addEventListener("click", resetView);
      zoomSlider.addEventListener("input", () => {
        cancelAnimationFrame(viewAnimation);
        setZoom(zoomFromSlider(Number(zoomSlider.value)));
      });

      // The map turns slowly by default. It pauses while the pointer is over
      // it (so dots hold still under the cursor), while you drag, and while
      // it is off screen.
      const tick = () => {
        if (autoRotateInput.checked && cloud.x && visible && !pointerOver
            && !drag) {
          view.yaw += 0.005;
          draw();
        }
      };

      return { method, setData, setSelection, resetView, draw, tick };
    };

    for (const panel of document.querySelectorAll(".map-panel")) {
      plots.push(createPlot(panel));
    }

    // Toggling the Cellpose highlight only changes colors, so a plain redraw
    // per map is enough; no data is re-queried.
    highlightMissedInput.addEventListener("change", () => {
      for (const plot of plots) {
        plot.draw();
      }
    });

    const redrawClusters = (
      seedObjectNumber,
      neighborObjectNumbers,
      hnswNeighborObjectNumbers,
    ) => {
      selection = {
        seed: seedObjectNumber ?? null,
        neighbors: neighborObjectNumbers ?? [],
        hnsw: hnswNeighborObjectNumbers ?? [],
      };
      for (const plot of plots) {
        plot.setSelection(selection);
      }
    };

    document.querySelector("#reset-all").addEventListener("click", () => {
      for (const plot of plots) {
        plot.resetView();
      }
    });

    // Rotation is off for people who ask for reduced motion.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      autoRotateInput.checked = false;
    }
    const spin = () => {
      for (const plot of plots) {
        plot.tick();
      }
      requestAnimationFrame(spin);
    };
    requestAnimationFrame(spin);

    const updateFeatureButtons = () => {
      for (const button of featureTabs.querySelectorAll("button")) {
        button.setAttribute(
          "aria-pressed",
          button.dataset.featureSpace === activeFeatureSpace ? "true" : "false",
        );
      }
    };

    const setFeatureSpace = async (featureSpace) => {
      if (!connection || featureSpace === activeFeatureSpace) {
        return;
      }
      activeFeatureSpace = featureSpace;
      updateFeatureButtons();
      setStatus(`Loading ${FEATURE_SPACES[featureSpace].label} map...`);
      await loadClusterData();
      const current = normalizeObjectNumber(objectNumberInput.value);
      await runSearch(current ?? DEFAULT_OBJECT_NUMBER);
    };

    for (const [featureSpace, feature] of Object.entries(FEATURE_SPACES)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "secondary";
      button.dataset.featureSpace = featureSpace;
      button.textContent = feature.label;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        setFeatureSpace(featureSpace);
      });
      featureTabs.append(button);
    }
    updateFeatureButtons();

    const renderSeed = (row) => {
      seedFigure.hidden = false;
      seedImage.src = jpegToDataUrl(row.pixel_data_jpeg);
      seedCaption.textContent = `nucleus ${cellValue(row.ObjectNumber)} (search seed)`;
    };

    const renderResults = (rows) => {
      resultsEl.replaceChildren();
      for (const row of rows) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "result";
        const img = document.createElement("img");
        img.src = jpegToDataUrl(row.pixel_data_jpeg);
        img.alt = `nucleus ${cellValue(row.ObjectNumber)}`;
        const meta = document.createElement("div");
        meta.className = "meta";
        const idLine = document.createElement("span");
        idLine.textContent = `nucleus ${cellValue(row.ObjectNumber)}`;
        const distanceLine = document.createElement("span");
        distanceLine.className = "distance";
        distanceLine.textContent = `distance ${Number(row.dist).toFixed(3)}`;
        meta.append(idLine, distanceLine);
        button.append(img, meta);
        button.addEventListener("click", () => runSearch(cellValue(row.ObjectNumber)));
        resultsEl.append(button);
      }
    };

    const runSearch = async (objectNumber) => {
      objectNumber = normalizeObjectNumber(objectNumber);
      if (!objectNumber) {
        setStatus("Enter a whole nucleus number.");
        return;
      }
      searchButton.disabled = true;
      randomButton.disabled = true;
      try {
        const feature = FEATURE_SPACES[activeFeatureSpace];
        setStatus(`Finding nuclei similar to nucleus ${objectNumber}...`);
        const seedTable = await connection.query(`
          SELECT oc.ObjectNumber, oc.pixel_data_jpeg
          FROM database.images.object_crops oc
          WHERE oc.object_type = 'nucleus' AND oc.channel = 'DNA'
              AND oc.ObjectNumber = ${objectNumber}
          LIMIT 1
        `);
        const seedRows = rowsOf(seedTable);
        if (seedRows.length === 0) {
          setStatus(`No nucleus found with number ${objectNumber}.`);
          resultsEl.replaceChildren();
          seedFigure.hidden = true;
          return;
        }
        renderSeed(seedRows[0]);

        // DuckDB uses an HNSW index only when the query vector is a
        // constant, so copy the seed vector into a variable first. The
        // index answers the ORDER BY ... LIMIT inside the CTE, and only its
        // 13 hits are joined to the crop images.
        await connection.query(`
          SET VARIABLE query_vector = (
              SELECT ${feature.vectorColumn} FROM ${feature.table}
              WHERE ObjectNumber = ${objectNumber}
          )
        `);
        const neighborTable = await connection.query(`
          WITH nn AS (
              SELECT
                  ObjectNumber,
                  array_distance(
                      ${feature.vectorColumn},
                      getvariable('query_vector')::FLOAT[${feature.dim}]
                  ) AS dist
              FROM ${feature.table}
              ORDER BY dist
              LIMIT 13
          )
          SELECT nn.ObjectNumber, oc.pixel_data_jpeg, nn.dist
          FROM nn
          JOIN database.images.object_crops oc
              ON oc.ObjectNumber = nn.ObjectNumber
              AND oc.object_type = 'nucleus' AND oc.channel = 'DNA'
          WHERE nn.ObjectNumber != ${objectNumber}
          ORDER BY nn.dist
          LIMIT 12
        `);
        const neighborRows = rowsOf(neighborTable).map((row) => (
          { ...row, dist: cellValue(row.dist) }
        ));
        renderResults(neighborRows);
        objectNumberInput.value = objectNumber;
        redrawClusters(
          String(objectNumber),
          neighborRows.map((row) => String(cellValue(row.ObjectNumber))),
          neighborRows.slice(0, 6).map((row) => (
            String(cellValue(row.ObjectNumber))
          )),
        );
        setStatus(
          `Showing similar nuclei from the ${feature.label} feature space `
          + (hnswAvailable && feature.indexed
            ? "(DuckDB HNSW index)."
            : "(exact scan)."),
        );
      } catch (error) {
        setStatus(`Search failed: ${error.message}`);
      } finally {
        searchButton.disabled = false;
        randomButton.disabled = false;
      }
    };

    const pickRandomObjectNumber = async () => {
      const table = await connection.query(`
        SELECT ObjectNumber
        FROM database.features.fused
        ORDER BY random()
        LIMIT 1
      `);
      return cellValue(rowsOf(table)[0].ObjectNumber);
    };

    const sqlEditor = document.querySelector("#sql-editor");
    const sqlExamplesEl = document.querySelector("#sql-examples");
    const sqlButton = document.querySelector("#run-sql");
    const sqlStatusEl = document.querySelector("#sql-status");
    const sqlResultsEl = document.querySelector("#sql-results");

    const sqlTableRows = (table) => {
      const fields = table.schema.fields.map((field) => field.name);
      const rows = [];
      for (let rowIndex = 0; rowIndex < table.numRows; rowIndex += 1) {
        const row = {};
        fields.forEach((fieldName, fieldIndex) => {
          const column = table.getChild?.(fieldName) ?? table.getChildAt(fieldIndex);
          row[fieldName] = sqlCellValue(column.get(rowIndex));
        });
        rows.push(row);
      }
      return { fields, rows };
    };

    const renderSqlTable = (table) => {
      const { fields, rows } = sqlTableRows(table);
      sqlResultsEl.replaceChildren();
      if (fields.length === 0) {
        const message = document.createElement("p");
        message.textContent = "Query returned no columns.";
        sqlResultsEl.append(message);
        return;
      }
      const tableEl = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const field of fields) {
        const th = document.createElement("th");
        th.textContent = field;
        headerRow.append(th);
      }
      thead.append(headerRow);

      const tbody = document.createElement("tbody");
      for (const row of rows) {
        const tr = document.createElement("tr");
        for (const field of fields) {
          const td = document.createElement("td");
          td.textContent = row[field];
          tr.append(td);
        }
        tbody.append(tr);
      }
      tableEl.append(thead, tbody);
      sqlResultsEl.append(tableEl);
    };

    // the actual example queries from this project's README: real SQL
    // over the real tables, not placeholders.
    const SQL_EXAMPLES = [
      {
        label: "nuclei",
        sql: `SELECT
    ObjectNumber,
    image_id,
    AreaShape_Area,
    Intensity_MeanIntensity_DNA
FROM database.bbbc039.objects
LIMIT 10;`,
      },
      {
        label: "HNSW search",
        sql: `SET VARIABLE query_vector = (
    SELECT fused_embedding FROM database.features.fused WHERE ObjectNumber = 1
);
SELECT ObjectNumber,
       array_distance(fused_embedding,
                      getvariable('query_vector')::FLOAT[48]) AS distance
FROM database.features.fused
ORDER BY distance
LIMIT 10;`,
      },
      {
        label: "array_distance",
        sql: `SELECT ObjectNumber, array_distance(
    fused_embedding,
    (SELECT fused_embedding FROM database.features.fused WHERE ObjectNumber = 1)
) AS distance
FROM database.features.fused
ORDER BY distance
LIMIT 10;`,
      },
      {
        label: "knn_graph",
        sql: `SELECT ObjectNumber, neighbor_object_number, rank, distance
FROM database.features.knn_graph
WHERE feature_space = 'fused' AND ObjectNumber = 1
ORDER BY rank;`,
      },
      {
        label: "object_crops",
        sql: `SELECT ObjectNumber, image_id, size_y, size_x, height, width,
       octet_length(pixel_data_jpeg) AS jpeg_bytes
FROM database.images.object_crops
LIMIT 10;`,
      },
      {
        label: "images",
        sql: `SELECT ImageNumber, image_id, split, well, object_count,
       height, width
FROM database.images.images
ORDER BY ImageNumber
LIMIT 10;`,
      },
      {
        label: "Row counts",
        sql: `SELECT 'bbbc039.objects' AS table_name, COUNT(*) AS rows
FROM database.bbbc039.objects
UNION ALL
SELECT 'features.fused', COUNT(*) FROM database.features.fused
UNION ALL
SELECT 'features.knn_graph', COUNT(*) FROM database.features.knn_graph
UNION ALL
SELECT 'images.object_crops', COUNT(*) FROM database.images.object_crops;`,
      },
    ];

    for (const example of SQL_EXAMPLES) {
      const exampleButton = document.createElement("button");
      exampleButton.type = "button";
      exampleButton.className = "example";
      exampleButton.textContent = example.label;
      exampleButton.addEventListener("click", () => {
        sqlEditor.value = example.sql;
        runSql();
      });
      sqlExamplesEl.append(exampleButton);
    }
    sqlEditor.value = SQL_EXAMPLES[0].sql;

    const runSql = async () => {
      const sql = sqlEditor.value.trim();
      if (!sql || !connection) {
        return;
      }
      sqlButton.disabled = true;
      sqlResultsEl.replaceChildren();
      try {
        sqlStatusEl.textContent = "Running query...";
        const table = await connection.query(sql);
        renderSqlTable(table);
        sqlStatusEl.textContent = `Query complete: ${table.numRows} row(s).`;
      } catch (error) {
        sqlStatusEl.textContent = `Query failed: ${error.message}`;
      } finally {
        sqlButton.disabled = false;
      }
    };

    sqlButton.addEventListener("click", runSql);
    sqlEditor.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        runSql();
      }
    });

    const pickerEl = document.querySelector("#picker");
    const pickerInput = document.querySelector("#picker-input");

    const init = async (loadBytes = fetchJpegBytes) => {
      pickerEl.hidden = true;
      try {
        connection = await openDatabase(await loadBytes());
        searchButton.disabled = false;
        randomButton.disabled = false;
        sqlButton.disabled = false;
        sqlStatusEl.textContent = "Ready.";
        setStatus("Loading cluster maps...");
        await loadClusterData();
        redrawClusters(null, []);
        setStatus("Ready.");
        await runSearch(DEFAULT_OBJECT_NUMBER);
      } catch (error) {
        setStatus(`Failed to load database.jpg: ${error.message}`);
        // a page opened from disk (file://) cannot fetch() its neighbours;
        // let the person hand the same JPEG to the page directly instead.
        pickerEl.hidden = false;
      }
    };

    pickerInput.addEventListener("change", () => {
      const file = pickerInput.files[0];
      if (file) {
        init(() => file.arrayBuffer());
      }
    });

    searchButton.addEventListener("click", () => {
      const value = objectNumberInput.value.trim();
      if (value) {
        runSearch(value);
      }
    });
    objectNumberInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        searchButton.click();
      }
    });
    randomButton.addEventListener("click", async () => {
      randomButton.disabled = true;
      try {
        await runSearch(await pickRandomObjectNumber());
      } finally {
        randomButton.disabled = false;
      }
    });

    init();
  </script>
</body>
</html>
""".replace("__DEFAULT_FEATURE_SPACE__", DEFAULT_FEATURE_SPACE)
        .replace("__IMAGE_COUNT__", str(counts["bbbc039_images"]))
        .replace("__NUCLEUS_COUNT__", f"{counts['bbbc039_objects']:,}")
        .replace("__TRAINING_IMAGES__", str(counts["training_images"]))
        .replace("__VALIDATION_IMAGES__", str(counts["validation_images"]))
        .replace("__TEST_IMAGES__", str(counts["test_images"]))
        .replace("__FEATURE_SPACES__", "3")
        .replace(
            "__HNSW_RECALL__",
            f"{math.floor(min(counts['hnsw_recall'].values()) * 100)}%",
        )
        .replace("__REPO_URL__", REPO_URL)
        .replace("__LOGO_BLUE__", LOGO_BLUE)
        .replace("__LOGO_PINK__", LOGO_PINK)
        .replace("__MISSED_COLOR__", MISSED_COLOR)
        .replace("__JPEG_MB__", f"{counts['jpeg_bytes'] / 1e6:.0f} MB")
        .replace("__DENSITY_STOPS__", json.dumps([list(c) for c in DENSITY_STOPS]))
        .replace(
            "__DENSITY_GRADIENT__",
            ", ".join(f"rgb({r}, {g}, {b})" for r, g, b in DENSITY_STOPS),
        )
        .replace("__CP_MEASURE_URL__", CP_MEASURE_URL)
        .replace("__CP_MEASURE_PAPER_URL__", CP_MEASURE_PAPER_URL)
        .replace("__CP_MEASURE_VERSION__", CP_MEASURE_VERSION)
        .replace(
            "__CELLPOSE_CAPTURE_PCT__",
            str(counts["cellpose_capture_pct"]).rstrip("0").rstrip("."),
        )
        .replace(
            "__CELLPOSE_MISSED_COUNT__",
            f"{counts['bbbc039_objects'] - counts['cellpose_captured']:,}",
        )
        .replace(
            "__CELLPOSE_MISSED_PCT__",
            f"{100 - counts['cellpose_capture_pct']:.1f}".rstrip("0").rstrip("."),
        )
    )
    for placeholder, svg in all_diagrams(
        counts["jpeg_bytes"],
        CP_MEASURE_DIM,
        FUSED_EMBEDDING_DIM - CP_MEASURE_DIM,
        FUSED_EMBEDDING_DIM,
    ).items():
        html = html.replace(placeholder, svg)
    path.write_text(html, encoding="utf-8")
