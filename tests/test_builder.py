from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import duckdb
import pytest
from PIL import Image

from picture_thousand_records import build_artifacts
from picture_thousand_records.builder import (
    CELLPOSE_VERSION,
    CP_MEASURE_DIM,
    CP_MEASURE_VERSION,
    FUSED_EMBEDDING_DIM,
    SCHEMA_TREE_COL_A,
    SCHEMA_TREE_COL_B,
    ArtifactPaths,
)
from picture_thousand_records.diagrams import all_diagrams
from picture_thousand_records.duckdb_access import connect_to_database
from picture_thousand_records.validation import validate_artifact


@pytest.fixture(scope="module")
def built_paths(tmp_path_factory: pytest.TempPathFactory) -> ArtifactPaths:
    return build_artifacts(tmp_path_factory.mktemp("artifact"))


def test_build_artifact_contains_jpeg_zip_and_duckdb(
    built_paths: ArtifactPaths,
) -> None:
    with Image.open(built_paths.jpeg) as image:
        assert image.format == "JPEG"
        assert image.size == (1440, 1440)

    with zipfile.ZipFile(built_paths.jpeg) as archive:
        assert set(archive.namelist()) == {
            "database.duckdb",
            "README.md",
        }

    report = validate_artifact(built_paths.jpeg)
    counts = report["counts"]
    assert counts["bbbc039_images"] == 200
    assert counts["bbbc039_objects"] == 19565
    assert counts["bbbc039_annotations"] == 19565
    assert counts["bbbc039_segmentations"] == 200
    assert counts["images_metadata"] == 200
    assert counts["images_object_crops"] == 19565
    assert counts["features_cp_measure"] == 19565
    assert counts["features_morphem"] == 19565
    assert counts["features_fused"] == 19565
    assert counts["features_knn_graph"] == 19565 * 6 * 3
    assert report["duckdb_access"] in {"zipfs", "extracted"}
    assert report["manifest"]["schemas"] == ["bbbc039", "images", "features"]


def test_bbbc039_tables_hold_full_dataset(built_paths: ArtifactPaths) -> None:
    attached = connect_to_database(
        built_paths.jpeg,
        temp_dir=built_paths.output_dir / "x",
    )
    try:
        con = attached.connection
        split_counts = dict(
            con.sql(
                """
                SELECT split, COUNT(*)
                FROM database.bbbc039.images
                GROUP BY split
                """
            ).fetchall()
        )
        object_sum, median_nuclei, max_nuclei = con.sql(
            """
            SELECT SUM(object_count), median(object_count), max(object_count)
            FROM database.bbbc039.images
            """
        ).fetchone()
        sample = con.sql(
            """
            SELECT ObjectNumber, image_id, AreaShape_Area,
                   Intensity_MeanIntensity_DNA
            FROM database.bbbc039.objects
            ORDER BY ObjectNumber
            LIMIT 5
            """
        ).fetchall()
    finally:
        attached.connection.close()

    assert split_counts == {"training": 100, "validation": 50, "test": 50}
    assert object_sum == 19565
    assert median_nuclei == 105
    assert max_nuclei == 161
    assert len(sample) == 5
    assert all(row[2] > 0 and row[3] > 0 for row in sample)


def test_database_stores_no_original_images_or_masks(
    built_paths: ArtifactPaths,
) -> None:
    attached = connect_to_database(
        built_paths.jpeg,
        temp_dir=built_paths.output_dir / "x2",
    )
    try:
        con = attached.connection
        tables = {
            f"{schema}.{name}"
            for schema, name in con.sql(
                """
                SELECT table_schema, table_name FROM information_schema.tables
                WHERE table_catalog = 'database'
                """
            ).fetchall()
        }
        blob_columns = con.sql(
            """
            SELECT table_schema || '.' || table_name, column_name
            FROM information_schema.columns
            WHERE table_catalog = 'database' AND data_type = 'BLOB'
            """
        ).fetchall()
        images = con.sql(
            """
            SELECT source_filename, mask_filename, height, width, size_c, size_z,
                   size_y, size_x, dtype
            FROM database.images.images
            ORDER BY ImageNumber
            LIMIT 3
            """
        ).fetchall()
    finally:
        attached.connection.close()

    assert "images.source_images" not in tables
    # the only image bytes in the database are the JPEG crops
    assert blob_columns == [("images.object_crops", "pixel_data_jpeg")]
    for row in images:
        source_filename, mask_filename, *geometry, dtype = row
        assert source_filename.endswith(".tif")
        assert mask_filename.endswith(".png")
        assert geometry == [520, 696, 1, 1, 520, 696]
        assert dtype == "uint16"

    manifest = json.loads(built_paths.manifest.read_text(encoding="utf-8"))
    assert manifest["source"]["crop_columns"] == ["pixel_data_jpeg"]
    assert "original TIFF images" in manifest["source"]["not_stored"]


def test_object_crops_are_self_describing_records(
    built_paths: ArtifactPaths,
) -> None:
    attached = connect_to_database(
        built_paths.jpeg,
        temp_dir=built_paths.output_dir / "x3",
    )
    try:
        con = attached.connection
        row = con.sql(
            """
            SELECT height, width, size_c, size_z, size_y, size_x, z,
                   source, source_url,
                   octet_length(pixel_data_jpeg) AS jpeg_bytes
            FROM database.images.object_crops
            WHERE object_type = 'nucleus'
            LIMIT 1
            """
        ).fetchone()
    finally:
        attached.connection.close()

    (
        height,
        width,
        size_c,
        size_z,
        size_y,
        size_x,
        z,
        source,
        source_url,
        jpeg_bytes,
    ) = row
    assert (size_c, size_z, size_y, size_x, z) == (1, 1, 520, 696, 0)
    assert source == "Broad Bioimage Benchmark Collection BBBC039"
    assert source_url == "https://bbbc.broadinstitute.org/BBBC039"
    assert height > 0 and width > 0
    assert jpeg_bytes > 0


def test_feature_spaces_hold_fixed_size_vectors_and_real_neighbors(
    built_paths: ArtifactPaths,
) -> None:
    attached = connect_to_database(
        built_paths.jpeg,
        temp_dir=built_paths.output_dir / "x4",
    )
    try:
        con = attached.connection
        dims = {
            "cp_measure": con.sql(
                "SELECT embedding_dim FROM database.features.cp_measure LIMIT 1"
            ).fetchone()[0],
            "morphem": con.sql(
                "SELECT embedding_dim FROM database.features.morphem LIMIT 1"
            ).fetchone()[0],
            "fused": con.sql(
                "SELECT embedding_dim FROM database.features.fused LIMIT 1"
            ).fetchone()[0],
        }
        distance = con.sql(
            """
            SELECT array_distance(fused_embedding, fused_embedding)
            FROM database.features.fused
            LIMIT 1
            """
        ).fetchone()[0]
        graph_counts = dict(
            con.sql(
                """
                SELECT feature_space, COUNT(*)
                FROM database.features.knn_graph
                GROUP BY feature_space
                """
            ).fetchall()
        )
        distinct_umap = con.sql(
            "SELECT COUNT(DISTINCT tsne_z) FROM database.features.fused"
        ).fetchone()[0]
    finally:
        attached.connection.close()

    assert dims == {"cp_measure": 16, "morphem": 384, "fused": 48}
    assert distance == 0
    assert graph_counts == {
        "cp_measure": 19565 * 6,
        "morphem": 19565 * 6,
        "fused": 19565 * 6,
    }
    assert distinct_umap > 19000


def test_browser_page_contains_bbbc039_interactive_page(
    built_paths: ArtifactPaths,
) -> None:
    html = built_paths.browser_page.read_text(encoding="utf-8")
    for expected in [
        "@duckdb/duckdb-wasm",
        "JSZip.loadAsync",
        'id="picker-input"',
        "BBBC039",
        "Dataset at a glance",
        'class="dataset-stats"',
        "19,565",
        "Feature spaces",
        'id="feature-tabs"',
        "cp_measure",
        "MorphEm",
        "Fused",
        "What does fusion show?",
        "Three ways to draw the same vectors",
        "What is neighbor density?",
        "What the picture carries",
        'class="page-footer"',
        "Back to top",
        'id="top"',
        'class="title-picture"',
        'class="title-records"',
        "Why use an index?",
        "Hierarchical Navigable",
        'href="database.jpg"',
        "How to Use t-SNE Effectively",
        'data-method="pca"',
        'data-method="umap"',
        'data-method="tsne"',
        "f.tsne_z",
        "f.pca_z",
        'href="https://github.com/d33bs/picture-thousand-records"',
        "Source on GitHub",
        'class="icon" aria-hidden="true">🖼️',
        "one image file can carry both its data",
        "Compare the nuclei in feature space",
        "Search for a nucleus by object number",
        "f.umap_z",
        'id="reset-all"',
        'id="auto-rotate" type="checkbox" checked',
        'class="plot-zoom"',
        "plot-zoom-selection",
        'addEventListener("wheel"',
        'addEventListener("pointermove"',
        "dense neighbors",
        "let cropImageCache = new Map();",
        "fetchCropImageDataUrl",
        "Crop images are fetched on demand",
        "database.bbbc039.objects",
        "o.cellpose_captured",
        "Highlight nuclei Cellpose missed",
        'id="highlight-missed" type="checkbox" checked',
        "missed by Cellpose",
        "intersection over union",
        "https://en.wikipedia.org/wiki/Jaccard_index",
        "https://doi.org/10.1186/gb-2006-7-10-r100",
        "https://doi.org/10.1038/s41592-020-01018-x",
        "https://doi.org/10.1101/2025.04.28.651001",
        "Cellpose-SAM",
        "How to read the Cellpose overlap",
        "feature vectors did not include the Cellpose result",
        "which side explains",
        'style="background:#c77dff"',
        'const MISSED_COLOR = "#c77dff";',
        "database.features.cp_measure",
        "database.features.morphem",
        "database.features.fused",
        "database.features.knn_graph",
        "database.images.object_crops",
        "AreaShape_Area",
        "Intensity_MeanIntensity_DNA",
        "fused_embedding",
        "measurement_vector",
        "array_distance(",
        'const DEFAULT_OBJECT_NUMBER = "269";',
        "await runSearch(DEFAULT_OBJECT_NUMBER);",
        'await conn.query("INSTALL vss");',
        "SET VARIABLE query_vector",
        "getvariable('query_vector')",
        'src="database.jpg"',
        'cache: "no-store"',
    ]:
        assert expected in html
    html_text = " ".join(html.split())
    assert "cannot decide which side explains a miss" in html_text
    assert "warehouse." not in html
    assert "AS warehouse" not in html
    assert "__NUCLEUS_COUNT__" not in html
    assert "__DEFAULT_FEATURE_SPACE__" not in html
    assert "__CELLPOSE_CAPTURE_PCT__" not in html
    assert "__CELLPOSE_MISSED_COUNT__" not in html
    assert "__CELLPOSE_MISSED_PCT__" not in html
    assert "HNSW k-NN edge" not in html
    assert "hnswlib" not in html
    assert "Moffat et al. 2006" not in html


def test_browser_page_contains_inline_svg_diagrams(
    built_paths: ArtifactPaths,
) -> None:
    html = built_paths.browser_page.read_text(encoding="utf-8")
    for expected in [
        "fig-polyglot",
        "fig-size",
        "fig-index",
        "fig-maps",
        "fig-density",
        "fig-fusion",
        "fig-not",
        "fig-carries",
    ]:
        assert expected in html
        assert f'{expected}-title' in html
    assert "__DIAGRAM_" not in html


def test_browser_diagram_placeholders_match_registry() -> None:
    builder_source = (
        Path(__file__).parent.parent / "src/picture_thousand_records/builder.py"
    ).read_text(encoding="utf-8")
    template_start = builder_source.index("html = (")
    template_end = builder_source.index(
        '""".replace("__DEFAULT_FEATURE_SPACE__"', template_start
    )
    placeholders = set(
        re.findall(r"__DIAGRAM_[A-Z_]+__", builder_source[template_start:template_end])
    )
    registry = all_diagrams(
        jpeg_bytes=1,
        cp_dim=CP_MEASURE_DIM,
        morphem_dim=FUSED_EMBEDDING_DIM - CP_MEASURE_DIM,
        fused_dim=FUSED_EMBEDDING_DIM,
    )
    title_ids = re.findall(r'<title id="([^"]+)">', "".join(registry.values()))

    assert placeholders == set(registry)
    assert all(svg.startswith('<svg class="fig"') for svg in registry.values())
    assert all('role="img"' in svg for svg in registry.values())
    assert len(title_ids) == len(set(title_ids)) == len(registry)


def test_documented_sql_recipe_runs_as_printed(built_paths: ArtifactPaths) -> None:
    readme = built_paths.readme.read_text(encoding="utf-8")
    blocks = re.findall(r"```sql\n(.*?)```", readme, flags=re.S)
    recipe = next(block for block in blocks if "read_blob" in block)
    run_dir = built_paths.output_dir / "user_folder"
    run_dir.mkdir()
    shutil.copy(built_paths.jpeg, run_dir / "database.jpg")

    con = duckdb.connect()
    statements = [
        statement.strip()
        for statement in re.sub(r"--.*", "", recipe).split(";")
        if statement.strip()
    ]
    here = run_dir.as_posix()
    try:
        for statement in statements:
            con.execute(
                statement.replace(
                    "'database.duckdb'", f"'{here}/database.duckdb'"
                ).replace("zip://database.jpg", f"zip://{here}/database.jpg")
            )
    except duckdb.IOException as error:
        pytest.skip(f"zipfs unavailable: {error}")
    count = con.sql("SELECT count(*) FROM database.bbbc039.objects").fetchone()
    assert count == (19565,)


def test_cp_measure_is_described_accurately(built_paths: ArtifactPaths) -> None:
    lock = (Path(__file__).parent.parent / "uv.lock").read_text(encoding="utf-8")
    locked = re.search(r'name = "cp-measure"\nversion = "([^"]+)"', lock)
    assert locked and locked.group(1) == CP_MEASURE_VERSION

    with zipfile.ZipFile(built_paths.jpeg) as archive:
        embedded_readme = archive.read("README.md").decode("utf-8")
    html = built_paths.browser_page.read_text(encoding="utf-8")
    manifest = built_paths.manifest.read_text(encoding="utf-8")
    for text in (embedded_readme, html):
        assert "github.com/afermg/cp_measure" in text
        assert "CellProfiler-derived" not in text
        assert "271" in text
    assert "arxiv.org/abs/2507.01163" in html
    assert "get_core_measurements(legacy=True)" in manifest
    assert "get_core_measurements(legacy=True)" in embedded_readme

    # the documented counts must match the stored table
    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.jpeg.parent / "extracted-cp"
    )
    try:
        con = attached.connection
        columns = [
            name
            for (name,) in con.sql(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'bbbc039' AND table_name = 'objects'
                """
            ).fetchall()
        ]
    finally:
        attached.connection.close()
    prefixes = (
        "AreaShape_",
        "Granularity_",
        "Intensity_",
        "RadialDistribution_",
        "Texture_",
    )
    assert sum(name.startswith(prefixes) for name in columns) == 263
    # cp_measure also returns the Location_* values; the build script adds
    # Location_Center_Z itself, so it is not one of the 271
    locations = [name for name in columns if name.startswith("Location_")]
    assert 263 + len(locations) - 1 == 271


def test_cellpose_capture_columns_match_manifest(
    built_paths: ArtifactPaths,
) -> None:
    lock = (Path(__file__).parent.parent / "uv.lock").read_text(encoding="utf-8")
    locked = re.search(r'name = "cellpose"\nversion = "([^"]+)"', lock)
    assert locked and locked.group(1) == CELLPOSE_VERSION

    manifest = json.loads(built_paths.manifest.read_text(encoding="utf-8"))
    cellpose = manifest["cellpose"]
    assert cellpose["library"] == "cellpose"
    assert cellpose["model"] == "cpsam_v2"
    assert cellpose["iou_threshold"] == 0.5

    with zipfile.ZipFile(built_paths.jpeg) as archive:
        embedded_readme = archive.read("README.md").decode("utf-8")
    assert "cellpose" in embedded_readme
    # Wording is line-wrapped in the template, so assert on a single line.
    assert "are not stored; re-run `tools/bbbc039_data/match_cellpose.py`" in (
        embedded_readme
    )
    readme_text = " ".join(embedded_readme.split())
    assert "IoU means intersection over union" in readme_text
    assert "https://en.wikipedia.org/wiki/Jaccard_index" in embedded_readme
    assert "Cellpose verdict was not part of the feature vector" in readme_text
    assert "which side explains a miss" in readme_text
    html = built_paths.browser_page.read_text(encoding="utf-8")
    assert "were captured by a Cellpose mask" in " ".join(html.split())

    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.jpeg.parent / "extracted-cellpose"
    )
    try:
        con = attached.connection
        kinds = dict(
            con.sql(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'bbbc039' AND table_name = 'objects'
                  AND column_name IN ('cellpose_captured', 'cellpose_iou')
                """
            ).fetchall()
        )
        assert kinds == {"cellpose_captured": "BOOLEAN", "cellpose_iou": "FLOAT"}
        captured, missed, nulls = con.sql(
            """
            SELECT
                COUNT(*) FILTER (WHERE cellpose_captured),
                COUNT(*) FILTER (WHERE NOT cellpose_captured),
                COUNT(*) FILTER (WHERE cellpose_captured IS NULL)
            FROM database.bbbc039.annotations
            """
        ).fetchone()
        # Every captured nucleus must have cleared the IoU threshold. A
        # missed nucleus can still show iou >= 0.5 when it competed for a
        # mask another nucleus matched better, so only check the one side.
        under = con.sql(
            """
            SELECT COUNT(*)
            FROM database.bbbc039.annotations
            WHERE cellpose_captured AND cellpose_iou < 0.5
            """
        ).fetchone()[0]
    finally:
        attached.connection.close()
    assert nulls == 0
    assert under == 0
    assert captured + missed == 19565
    assert captured == cellpose["captured_nuclei"]
    assert missed == cellpose["missed_nuclei"]
    assert cellpose["captured_pct"] == round(100 * captured / 19565, 1)


def test_hnsw_indexes_serve_the_search(built_paths: ArtifactPaths) -> None:
    manifest = json.loads(built_paths.manifest.read_text(encoding="utf-8"))
    recall = manifest["features"]["hnsw_index"]["recall_at_12_vs_exact_scan"]
    # MorphEm has no stored index (it would add about 33 MB)
    assert set(recall) == {"cp_measure", "fused"}
    assert manifest["features"]["hnsw_index"]["not_indexed"] == ["morphem"]
    assert min(recall.values()) >= 0.95

    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.output_dir / "x-hnsw"
    )
    try:
        if not attached.hnsw:
            pytest.skip("the vss extension could not be loaded")
        con = attached.connection
        indexed = dict(
            con.sql(
                "SELECT table_name, index_name FROM duckdb_indexes() "
                "WHERE index_name LIKE '%_hnsw'"
            ).fetchall()
        )
        assert indexed == {
            "cp_measure": "cp_measure_hnsw",
            "fused": "fused_hnsw",
        }

        # the same statements the page runs, in the same order
        con.execute(
            "SET VARIABLE query_vector = (SELECT fused_embedding "
            "FROM database.features.fused WHERE ObjectNumber = 269)"
        )
        page_query = """
            WITH nn AS (
                SELECT ObjectNumber,
                       array_distance(fused_embedding,
                                      getvariable('query_vector')::FLOAT[48]) AS dist
                FROM database.features.fused
                ORDER BY dist
                LIMIT 13
            )
            SELECT nn.ObjectNumber, nn.dist
            FROM nn
            JOIN database.images.object_crops oc
                ON oc.ObjectNumber = nn.ObjectNumber
                AND oc.object_type = 'nucleus' AND oc.channel = 'DNA'
            WHERE nn.ObjectNumber != 269
            ORDER BY nn.dist
            LIMIT 12
        """
        plan = " ".join(
            str(cell)
            for row in con.execute("EXPLAIN " + page_query).fetchall()
            for cell in row
        )
        assert "HNSW_INDEX_SCAN" in plan
        found = [row[0] for row in con.execute(page_query).fetchall()]

        # exact scan with the index switched off must agree closely
        con.execute("SET disabled_optimizers = 'extension'")
        exact = [
            row[0]
            for row in con.execute(
                """
                SELECT ObjectNumber FROM database.features.fused
                WHERE ObjectNumber != 269
                ORDER BY array_distance(
                    fused_embedding, getvariable('query_vector')::FLOAT[48])
                LIMIT 12
                """
            ).fetchall()
        ]
        con.execute("RESET disabled_optimizers")
    finally:
        attached.connection.close()

    assert len(found) == 12
    assert len(set(found) & set(exact)) >= 11
    assert found[0] == exact[0]


def test_database_stays_small(built_paths: ArtifactPaths) -> None:
    """The JPEG must stay under GitHub's 100 MB file limit."""

    assert built_paths.jpeg.stat().st_size < 100_000_000

    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.output_dir / "x-size"
    )
    try:
        con = attached.connection
        doubles = con.sql(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_catalog = 'database' AND table_schema = 'bbbc039'
              AND table_name = 'objects' AND data_type = 'DOUBLE'
            """
        ).fetchone()[0]
        floats = con.sql(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_catalog = 'database' AND table_schema = 'bbbc039'
              AND table_name = 'objects' AND data_type = 'FLOAT'
            """
        ).fetchone()[0]
        area = con.sql(
            "SELECT min(AreaShape_Area) FROM database.bbbc039.objects"
        ).fetchone()[0]
        con.execute(
            "SET VARIABLE query_vector = (SELECT embedding "
            "FROM database.features.morphem WHERE ObjectNumber = 1)"
        )
        morphem_plan = " ".join(
            str(cell)
            for row in con.execute(
                "EXPLAIN SELECT ObjectNumber FROM database.features.morphem "
                "ORDER BY array_distance(embedding, "
                "getvariable('query_vector')::FLOAT[384]) LIMIT 13"
            ).fetchall()
            for cell in row
        )
    finally:
        attached.connection.close()

    assert doubles == 0
    assert floats >= 263
    assert area > 0
    # MorphEm is searched by an exact scan, not an index
    assert "HNSW_INDEX_SCAN" not in morphem_plan


def test_cover_schema_matches_database(built_paths: ArtifactPaths) -> None:
    """The schema tree printed on the JPEG must list the real tables."""

    printed = {
        f"{schema}.{table}"
        for schema, tables in SCHEMA_TREE_COL_A + SCHEMA_TREE_COL_B
        for table in tables
    }
    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.output_dir / "x-schema"
    )
    try:
        actual = {
            f"{schema}.{name}"
            for schema, name in attached.connection.sql(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_catalog = 'database'"
            ).fetchall()
        }
    finally:
        attached.connection.close()
    assert printed == actual


def test_three_maps_and_their_quality(built_paths: ArtifactPaths) -> None:
    attached = connect_to_database(
        built_paths.jpeg, temp_dir=built_paths.output_dir / "x-maps"
    )
    try:
        con = attached.connection
        distinct = {
            (space, method): con.sql(
                f"SELECT COUNT(DISTINCT {method}_z) FROM database.features.{space}"
            ).fetchone()[0]
            for space in ("cp_measure", "morphem", "fused")
            for method in ("pca", "umap", "tsne")
        }
        quality = {
            (space, method): (recall, ratio, variance)
            for space, method, recall, ratio, variance in con.sql(
                """
                SELECT feature_space, method, neighbor_recall, distance_ratio,
                       explained_variance
                FROM database.features.embedding_quality
                """
            ).fetchall()
        }
    finally:
        attached.connection.close()

    # every nucleus has its own position on every map
    assert all(count > 19000 for count in distinct.values())
    assert len(quality) == 9
    for space in ("cp_measure", "morphem", "fused"):
        pca, umap, tsne = (quality[(space, m)] for m in ("pca", "umap", "tsne"))
        # the page tells readers to expect PCA lowest and t-SNE highest
        assert pca[0] < umap[0] < tsne[0]
        assert tsne[1] < umap[1] < pca[1]
        assert 0 < pca[2] < 1
        assert umap[2] is None  # only PCA has an explained variance


def test_page_states_the_real_file_size(built_paths: ArtifactPaths) -> None:
    html = built_paths.browser_page.read_text(encoding="utf-8")
    size = built_paths.jpeg.stat().st_size
    assert f"This file is {size / 1e6:.0f} MB." in html
    assert "__JPEG_MB__" not in html
    assert "Why keep it under 100 MB?" in html
    assert "GitHub blocks any file larger than" in html
    # the limit the page explains is GitHub's 100 MiB
    assert size < 100 * 1024 * 1024
