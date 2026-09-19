from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from picture_thousand_records import build_artifacts
from picture_thousand_records.duckdb_access import connect_to_warehouse
from picture_thousand_records.validation import validate_artifact


def test_build_artifact_contains_jpeg_zip_and_duckdb(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)

    with Image.open(paths.jpeg) as image:
        assert image.format == "JPEG"
        assert image.size == (1440, 1440)

    with zipfile.ZipFile(paths.jpeg) as archive:
        assert set(archive.namelist()) == {
            "warehouse.duckdb",
            "manifest.json",
            "README.md",
        }

    report = validate_artifact(paths.jpeg)
    counts = report["counts"]
    # this is a fixed, real, segmented microscopy field, so counts are exact
    assert counts["cellprofiler_nuclei"] == 274
    assert counts["cellprofiler_cells"] == 274
    assert counts["cellprofiler_cytoplasm"] == 262
    assert counts["cellprofiler_ph3"] == 28
    assert counts["images_metadata"] == 1
    assert counts["images_object_crops"] == 822
    assert counts["morphem_features"] == 274
    assert counts["morphem_knn_graph"] == 274 * 6
    assert report["duckdb_access"] in {"zipfs", "extracted"}
    assert report["manifest"]["schemas"] == ["cellprofiler", "images", "morphem"]
    alignment = report["manifest"]["morphem"]["mitotic_cluster_alignment"]
    assert 0 < alignment["mitotic_cells"] < alignment["total_cells"]
    # the real HNSW graph should cluster mitotic cells together far more
    # than chance -- otherwise this "alignment with the source paper" claim
    # embedded in index.html would be false
    purity_pct = alignment["mean_knn_neighbor_purity_pct"]
    baseline_pct = alignment["population_baseline_pct"]
    assert purity_pct > 3 * baseline_pct


def test_real_tables_hold_real_cellprofiler_features(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)
    attached = connect_to_warehouse(paths.jpeg, temp_dir=tmp_path / "extracted")
    try:
        con = attached.connection
        cells = con.sql(
            """
            SELECT ObjectNumber, AreaShape_Area, Parent_Nuclei
            FROM warehouse.cellprofiler.cells
            ORDER BY ObjectNumber
            LIMIT 5
            """
        ).fetchall()
        mismatched_cytoplasm_parents = con.sql(
            """
            SELECT COUNT(*)
            FROM warehouse.cellprofiler.cytoplasm
            WHERE Parent_Nuclei != Parent_Cells
            """
        ).fetchone()[0]
        valid_nuclei_parents = con.sql(
            """
            SELECT COUNT(*)
            FROM warehouse.cellprofiler.ph3 p
            JOIN warehouse.cellprofiler.nuclei n
                ON n.ObjectNumber = p.Parent_Nuclei
            """
        ).fetchone()[0]
        ph3_count = con.sql(
            "SELECT COUNT(*) FROM warehouse.cellprofiler.ph3"
        ).fetchone()[0]
    finally:
        attached.connection.close()

    assert len(cells) == 5
    assert all(area > 0 for _, area, _ in cells)
    assert all(object_number == parent for object_number, _, parent in cells)
    assert mismatched_cytoplasm_parents == 0
    assert valid_nuclei_parents == ph3_count


def test_single_cell_view_matches_cytotable_preset_join(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)
    attached = connect_to_warehouse(paths.jpeg, temp_dir=tmp_path / "extracted")
    try:
        con = attached.connection
        row_count = con.sql(
            "SELECT COUNT(*) FROM warehouse.cellprofiler.single_cell"
        ).fetchone()[0]
        cytoplasm_count = con.sql(
            "SELECT COUNT(*) FROM warehouse.cellprofiler.cytoplasm"
        ).fetchone()[0]
        row = con.sql(
            """
            SELECT Cytoplasm_ObjectNumber, Cells_ObjectNumber, Nuclei_ObjectNumber,
                   Cytoplasm_AreaShape_Area, Cells_AreaShape_Area, Nuclei_AreaShape_Area
            FROM warehouse.cellprofiler.single_cell
            ORDER BY Cytoplasm_ObjectNumber
            LIMIT 1
            """
        ).fetchone()
    finally:
        attached.connection.close()

    # CytoTable's cellprofiler_csv preset anchors this join on Cytoplasm and
    # LEFT JOINs Cells/Nuclei via Parent_Cells/Parent_Nuclei, so the view has
    # exactly one row per real cytoplasm object
    assert row_count == cytoplasm_count
    cytoplasm_id, cells_id, nuclei_id, cytoplasm_area, cells_area, nuclei_area = row
    assert cytoplasm_id == cells_id == nuclei_id
    assert cytoplasm_area > 0
    assert cells_area > 0
    assert nuclei_area > 0


def test_morphem_features_hold_real_embeddings(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)
    attached = connect_to_warehouse(paths.jpeg, temp_dir=tmp_path / "extracted")
    try:
        con = attached.connection
        model, channels, embedding_dim = con.sql(
            """
            SELECT model, channels, embedding_dim
            FROM warehouse.morphem.features
            LIMIT 1
            """
        ).fetchone()
        object_numbers = {
            row[0]
            for row in con.sql(
                "SELECT ObjectNumber FROM warehouse.morphem.features"
            ).fetchall()
        }
        cell_object_numbers = {
            row[0]
            for row in con.sql(
                "SELECT ObjectNumber FROM warehouse.cellprofiler.cells"
            ).fetchall()
        }
        distinct_embeddings = con.sql(
            "SELECT COUNT(DISTINCT embedding) FROM warehouse.morphem.features"
        ).fetchone()[0]
        self_distance = con.sql(
            """
            SELECT array_distance(embedding, embedding)
            FROM warehouse.morphem.features
            LIMIT 1
            """
        ).fetchone()[0]
        distinct_umap_x = con.sql(
            "SELECT COUNT(DISTINCT umap_x) FROM warehouse.morphem.features"
        ).fetchone()[0]
        distinct_umap_y = con.sql(
            "SELECT COUNT(DISTINCT umap_y) FROM warehouse.morphem.features"
        ).fetchone()[0]
    finally:
        attached.connection.close()

    assert model == "CaicedoLab/MorphEm"
    assert channels == ["DNA", "PH3", "cellbody"]
    assert embedding_dim == 1152
    # one real embedding per real cell, matching cellprofiler.cells exactly
    assert object_numbers == cell_object_numbers
    # real embeddings, not a placeholder repeated for every row
    assert distinct_embeddings == len(object_numbers)
    assert self_distance == 0
    # real UMAP projection of the embedding, not a constant placeholder
    assert distinct_umap_x == len(object_numbers)
    assert distinct_umap_y == len(object_numbers)


def test_knn_graph_holds_real_hnsw_neighbors(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)
    attached = connect_to_warehouse(paths.jpeg, temp_dir=tmp_path / "extracted")
    try:
        con = attached.connection
        distinct_sources, self_loops = con.sql(
            """
            SELECT COUNT(DISTINCT ObjectNumber), SUM(
                (ObjectNumber = neighbor_object_number)::INT
            )
            FROM warehouse.morphem.knn_graph
            """
        ).fetchone()
        edges_for_cell_1 = con.sql(
            "SELECT COUNT(*) FROM warehouse.morphem.knn_graph WHERE ObjectNumber = 1"
        ).fetchone()[0]
        hnsw_distance, exact_distance = con.sql(
            """
            SELECT g.distance, array_distance(f1.embedding, f2.embedding)
            FROM warehouse.morphem.knn_graph g
            JOIN warehouse.morphem.features f1 ON f1.ObjectNumber = g.ObjectNumber
            JOIN warehouse.morphem.features f2
                ON f2.ObjectNumber = g.neighbor_object_number
            WHERE g.ObjectNumber = 1 AND g.rank = 1
            """
        ).fetchone()
        distinct_graph_x = con.sql(
            "SELECT COUNT(DISTINCT graph_x) FROM warehouse.morphem.features"
        ).fetchone()[0]
    finally:
        attached.connection.close()

    assert distinct_sources == 274
    assert self_loops == 0
    assert edges_for_cell_1 == 6
    # a real HNSW index's approximate distance should closely match the
    # exact brute-force distance array_distance computes for the same pair
    assert abs(hnsw_distance - exact_distance) < 0.05
    # a real force-directed layout of the graph, not a constant placeholder
    assert distinct_graph_x == 274


def test_object_crops_are_self_describing_ome_arrow_records(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)
    attached = connect_to_warehouse(paths.jpeg, temp_dir=tmp_path / "extracted")
    try:
        con = attached.connection
        size_c, size_y, size_x = con.sql(
            "SELECT size_c, size_y, size_x FROM warehouse.images.images"
        ).fetchone()
        row = con.sql(
            """
            SELECT height, width, size_c, size_z, size_y, size_x, z,
                   source, source_url,
                   octet_length(pixel_data_raw) AS raw_bytes,
                   octet_length(pixel_data_jpegxl) AS jpegxl_bytes,
                   octet_length(pixel_data_jpeg) AS jpeg_bytes
            FROM warehouse.images.object_crops
            WHERE object_type = 'cells'
            LIMIT 1
            """
        ).fetchone()
        channel_count = con.sql(
            "SELECT COUNT(DISTINCT channel) FROM warehouse.images.object_crops"
        ).fetchone()[0]
    finally:
        attached.connection.close()

    (
        height,
        width,
        row_size_c,
        size_z,
        row_size_y,
        row_size_x,
        z,
        source,
        source_url,
        raw_bytes,
        jpegxl_bytes,
        jpeg_bytes,
    ) = row
    assert channel_count == 3
    # the crop row carries the source field's own OME-Arrow dimensions, so
    # no join to images.images is required to know what field it came from
    assert (row_size_c, row_size_y, row_size_x) == (size_c, size_y, size_x)
    assert size_z == 1
    assert z == 0
    assert source and source_url
    # pixel_data_raw is the uncompressed uint8 array (queryable with no
    # decoding); the other two are the same crop, compactly encoded.
    assert raw_bytes == height * width
    assert jpegxl_bytes > 0
    assert jpeg_bytes > 0


def test_browser_page_contains_wasm_adapter(tmp_path: Path) -> None:
    paths = build_artifacts(tmp_path)

    html = paths.browser_page.read_text(encoding="utf-8")
    assert "@duckdb/duckdb-wasm" in html
    assert "JSZip.loadAsync" in html
    assert 'zip.file("warehouse.duckdb")' in html
    assert 'registerFileBuffer("warehouse.duckdb"' in html
    assert "ATTACH 'warehouse.duckdb' AS warehouse (READ_ONLY)" in html
    assert "warehouse.images.object_crops" in html
    assert "warehouse.morphem.features" in html
    assert 'cache: "no-store"' in html
    assert 'src="database.jpg"' in html
    # an editable, re-runnable SQL console, preloaded with the actual
    # example queries this project's README documents
    assert 'id="sql-editor"' in html
    assert 'id="run-sql"' in html
    assert "warehouse.cellprofiler.single_cell" in html
    assert "warehouse.morphem.knn_graph" in html
    assert "array_distance(" in html
    assert "connection.query(sql)" in html
    # the interactive similarity search and plots, on the same page
    assert 'id="object-number"' in html
    assert 'id="search"' in html
    assert 'id="umap-canvas"' in html
    assert 'id="graph-canvas"' in html
    assert "umap_x" in html
    assert "umap_y" in html
    assert "graph_x" in html
    assert "graph_y" in html
    assert "UMAP" in html
    assert "HNSW" in html
    assert 'id="hover-tooltip"' in html
    assert "attachClusterInteraction" in html
    # figure captions (title via <h2>, canvas, then caption+legend below)
    assert "<figure" in html
    assert html.index("<h2>UMAP") < html.index('id="umap-canvas"')
    assert html.index('id="umap-canvas"') < html.index("<figcaption>")
    # a real, build-time-computed check against the dataset's source paper
    assert "Moffat et al. 2006" in html
    assert "__MITOTIC_CELLS__" not in html
    assert "__PURITY_PCT__" not in html
