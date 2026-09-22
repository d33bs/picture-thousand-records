"""Build the JPEG Warehouse polyglot artifact.

Every table in this warehouse is real: real CellProfiler-style
compartment tables (`cellprofiler.nuclei`/`cells`/`cytoplasm`/`ph3`), one-row
image metadata (`images.images`), and real per-object image crops
(`images.object_crops`), each crop a self-describing OME-Arrow-style record
carrying the source field's own dimensions alongside its own pixel data, all
measured and cropped from CellProfiler's public `ExampleHuman` tutorial field.
Cellpose, through its current default Cellpose-SAM model, supplies the
cell-body masks used for cells and crops. See
`tools/real_example_data/extract_features.py` for how those tables are
produced; this module only loads the resulting bundled parquet assets into a
fresh DuckDB database and wraps it in the JPEG/ZIP polyglot.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).parent / "assets"
SOURCE_URL = (
    "https://github.com/cytomining/CytoTable/tree/main/tests/"
    "data/cellprofiler/ExampleHuman"
)
MORPHEM_MODEL = "CaicedoLab/MorphEm"
MORPHEM_EMBEDDING_DIM = 1152


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths produced by a warehouse build."""

    output_dir: Path
    jpeg: Path
    duckdb: Path
    manifest: Path
    readme: Path
    browser_page: Path


def build_artifacts(output_dir: str | Path = ".") -> ArtifactPaths:
    """Build a JPEG/ZIP polyglot containing a DuckDB warehouse of real
    CellProfiler `ExampleHuman` data."""

    paths = _paths(Path(output_dir))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _unlink_outputs(paths)
    counts = _build_duckdb(paths.duckdb)
    manifest = _manifest(counts)
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths.readme.write_text(_embedded_readme(counts), encoding="utf-8")
    _write_cover_jpeg(paths.jpeg)
    _append_zip(paths)
    _write_browser_page(paths.browser_page, counts)
    return paths


def _paths(output_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        output_dir=output_dir,
        jpeg=output_dir / "database.jpg",
        duckdb=output_dir / "warehouse.duckdb",
        manifest=output_dir / "manifest.json",
        readme=output_dir / "WAREHOUSE_README.md",
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
        con.execute("CREATE SCHEMA cellprofiler")
        con.execute("CREATE SCHEMA images")
        con.execute("CREATE SCHEMA morphem")

        counts = {}
        for table_name in ["nuclei", "cells", "cytoplasm", "ph3"]:
            counts[table_name] = _load_parquet_table(
                con, f"real_{table_name}.parquet", f"cellprofiler.{table_name}"
            )
        _create_single_cell_view(con)
        counts["images_metadata"] = _load_parquet_table(
            con, "real_images_metadata.parquet", "images.images"
        )
        counts["images_object_crops"] = _load_parquet_table(
            con, "real_object_crops.parquet", "images.object_crops"
        )
        counts["morphem_features"] = _create_morphem_table(con)
        counts["morphem_knn_graph"] = _load_parquet_table(
            con, "real_morphem_knn_graph.parquet", "morphem.knn_graph"
        )
        counts.update(_mitotic_cluster_alignment(con, counts["cells"]))
        return counts
    finally:
        con.close()


def _mitotic_cluster_alignment(
    con: duckdb.DuckDBPyConnection,
    total_cells: int,
) -> dict[str, int]:
    """Check, from the real data alone, whether the real MorphEm/HNSW
    clustering recovers a real biological label: PH3-positive (mitotic)
    cells, the exact phenotype the `ExampleHuman` field's source screen
    (Moffat et al., 2006) used as its readout ("mitotic index"). Computed
    at build time, from the real tables, so it can never go stale.
    """

    mitotic_cells = con.sql(
        "SELECT COUNT(DISTINCT Parent_Cells) FROM cellprofiler.ph3"
    ).fetchone()[0]
    purity = con.sql(
        """
        WITH mitotic AS (
            SELECT DISTINCT Parent_Cells AS ObjectNumber FROM cellprofiler.ph3
        ),
        per_cell_purity AS (
            SELECT
                g.ObjectNumber,
                AVG(
                    (g.neighbor_object_number IN (SELECT ObjectNumber FROM mitotic))
                    ::DOUBLE
                ) AS neighbor_purity
            FROM morphem.knn_graph g
            WHERE g.ObjectNumber IN (SELECT ObjectNumber FROM mitotic)
            GROUP BY g.ObjectNumber
        )
        SELECT AVG(neighbor_purity) FROM per_cell_purity
        """
    ).fetchone()[0]
    baseline = mitotic_cells / total_cells
    return {
        "mitotic_cells": mitotic_cells,
        "mitotic_knn_purity_pct": round(purity * 100),
        "mitotic_baseline_pct": round(baseline * 100),
    }


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


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    return [
        row[0]
        for row in con.sql(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'cellprofiler' AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """
        ).fetchall()
    ]


def _create_single_cell_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create a merged single-cell view, following the same join CytoTable's
    `cellprofiler_csv` preset uses for this exact dataset in its own tests
    (https://github.com/cytomining/CytoTable/blob/main/cytotable/presets.py):
    anchor on Cytoplasm and LEFT JOIN Cells/Nuclei via the Parent_Cells/
    Parent_Nuclei foreign keys.

    CytoTable's preset SQL assumes its input parquet files already carry
    compartment-prefixed column names (e.g. `Cells_AreaShape_Area`), which it
    produces in an earlier ingestion step before that join ever runs. Our
    tables keep the raw CellProfiler-style names instead (to match
    `Cells.csv`/`Nuclei.csv`/`Cytoplasm.csv` directly), so joining them as-is
    would produce ambiguous duplicate column names -- all three compartments
    share the same ~97 measurement names. This generates that same
    Compartment_Measurement prefixing here, at view-creation time, so the
    result matches what a real CytoTable single-cell table looks like.
    """

    nuclei_cols = [
        c
        for c in _table_columns(con, "nuclei")
        if c not in ("ImageNumber", "ObjectNumber")
    ]
    cells_cols = [
        c
        for c in _table_columns(con, "cells")
        if c not in ("ImageNumber", "ObjectNumber", "Parent_Nuclei")
    ]
    cytoplasm_cols = [
        c
        for c in _table_columns(con, "cytoplasm")
        if c not in ("ImageNumber", "ObjectNumber", "Parent_Nuclei", "Parent_Cells")
    ]

    select_list = [
        "cytoplasm.ImageNumber",
        "cytoplasm.ObjectNumber AS Cytoplasm_ObjectNumber",
        "cytoplasm.Parent_Cells",
        "cytoplasm.Parent_Nuclei",
        *(f'cytoplasm."{c}" AS "Cytoplasm_{c}"' for c in cytoplasm_cols),
        "cells.ObjectNumber AS Cells_ObjectNumber",
        *(f'cells."{c}" AS "Cells_{c}"' for c in cells_cols),
        "nuclei.ObjectNumber AS Nuclei_ObjectNumber",
        *(f'nuclei."{c}" AS "Nuclei_{c}"' for c in nuclei_cols),
    ]
    con.execute(
        f"""
        CREATE VIEW cellprofiler.single_cell AS
        SELECT {", ".join(select_list)}
        FROM cellprofiler.cytoplasm AS cytoplasm
        LEFT JOIN cellprofiler.cells AS cells
            ON cells.ObjectNumber = cytoplasm.Parent_Cells
        LEFT JOIN cellprofiler.nuclei AS nuclei
            ON nuclei.ObjectNumber = cytoplasm.Parent_Nuclei
        """
    )


def _create_morphem_table(con: duckdb.DuckDBPyConnection) -> int:
    """Load real MorphEm embeddings, one 1152-d fixed-size array per real
    cell (384-d per channel from the real, pretrained
    `CaicedoLab/MorphEm` DINO Vision Transformer, concatenated across the
    DNA/PH3/cell-body channels), plus a real 2-D UMAP projection of that
    embedding (`umap_x`/`umap_y`) carried straight through from the parquet
    asset via `* EXCLUDE (embedding)` -- see
    `tools/real_example_data/extract_morphem_embeddings.py`.
    """

    asset_path = ASSETS_DIR / "real_morphem_features.parquet"
    con.execute(
        f"""
        CREATE TABLE morphem.features AS
        SELECT
            * EXCLUDE (embedding),
            CAST(embedding AS FLOAT[{MORPHEM_EMBEDDING_DIM}]) AS embedding
        FROM read_parquet('{asset_path.as_posix()}')
        """
    )
    return con.sql("SELECT COUNT(*) FROM morphem.features").fetchone()[0]


def _manifest(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "format": "jpeg-warehouse",
        "version": "0.2",
        "title": "Real CellProfiler ExampleHuman Warehouse",
        "database": "warehouse.duckdb",
        "database_engine": "duckdb",
        "access": {
            "filesystem": "zipfs",
            "member": "warehouse.duckdb",
            "split": "!!",
        },
        "schemas": ["cellprofiler", "images", "morphem"],
        "rows": {
            "cellprofiler.nuclei": counts["nuclei"],
            "cellprofiler.cells": counts["cells"],
            "cellprofiler.cytoplasm": counts["cytoplasm"],
            "cellprofiler.ph3": counts["ph3"],
            "images.images": counts["images_metadata"],
            "images.object_crops": counts["images_object_crops"],
            "morphem.features": counts["morphem_features"],
            "morphem.knn_graph": counts["morphem_knn_graph"],
        },
        "source": {
            "dataset": "CellProfiler ExampleHuman tutorial dataset (CC-0)",
            "source_url": SOURCE_URL,
            "cell_segmentation": "Cellpose default model on the cell-body channel",
            "cell_segmentation_reference": (
                "Stringer et al. 2021; Pachitariu & Stringer 2022; "
                "Pachitariu et al. 2025"
            ),
            "nuclei_and_ph3_segmentation": (
                "CellProfiler-style thresholding and intensity watershed"
            ),
            "measured_with": "cp_measure",
            "pipeline": "tools/real_example_data/ExampleHuman.cppipe",
            "crop_columns": ["pixel_data_raw", "pixel_data_jpegxl", "pixel_data_jpeg"],
        },
        "morphem": {
            "model": MORPHEM_MODEL,
            "model_url": f"https://huggingface.co/{MORPHEM_MODEL}",
            "embedding_dim": MORPHEM_EMBEDDING_DIM,
            "embedding_dim_per_channel": MORPHEM_EMBEDDING_DIM // 3,
            "channels": ["DNA", "PH3", "cellbody"],
            "projection": {
                "method": "UMAP",
                "library": "umap-learn",
                "reference": "McInnes, Healy & Melville (2018)",
                "metric": "euclidean",
                "random_state": 42,
                "columns": ["umap_x", "umap_y"],
            },
            "knn_graph": {
                "method": "HNSW k-NN graph",
                "library": "hnswlib",
                "reference": "Malkov & Yashunin (2016)",
                "k": 6,
                "metric": "euclidean",
                "edge_table": "morphem.knn_graph",
                "layout": {
                    "method": "Fruchterman-Reingold force-directed layout",
                    "library": "networkx",
                    "random_state": 42,
                    "columns": ["graph_x", "graph_y"],
                },
            },
            "mitotic_cluster_alignment": {
                "phenotype": "PH3-positive (mitotic) cells",
                "source_paper": "Moffat et al. 2006 (Cell)",
                "note": (
                    "the ExampleHuman field's source screen used PH3 "
                    "staining to score mitotic index; these unsupervised "
                    "embeddings recover the same distinction"
                ),
                "mitotic_cells": counts["mitotic_cells"],
                "total_cells": counts["cells"],
                "mean_knn_neighbor_purity_pct": counts["mitotic_knn_purity_pct"],
                "population_baseline_pct": counts["mitotic_baseline_pct"],
            },
        },
        "views": {
            "cellprofiler.single_cell": (
                "merged Cytoplasm/Cells/Nuclei single-cell profile, joined "
                "with the CytoTable-style parent-key pattern (anchored on "
                "Cytoplasm, LEFT JOIN Cells/Nuclei via "
                "Parent_Cells/Parent_Nuclei)"
            ),
        },
    }


def _embedded_readme(counts: dict[str, int]) -> str:
    return f"""# JPEG Warehouse

This JPEG contains an embedded ZIP archive with a DuckDB database.

DuckDB access (run `duckdb` in the folder that holds `database.jpg`):

```sql
INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';

-- ATTACH cannot open a zip:// path directly (DuckDB 1.5.x), so stream the
-- stored member out with read_blob, then attach the copy.
COPY (SELECT content FROM read_blob('zip://database.jpg!!warehouse.duckdb'))
    TO 'warehouse.duckdb' (FORMAT blob);

ATTACH 'warehouse.duckdb' AS warehouse (READ_ONLY);
```

To keep everything in memory afterwards, copy the tables and drop the file:

```sql
CREATE SCHEMA cellprofiler; CREATE SCHEMA images; CREATE SCHEMA morphem;
CREATE TABLE cellprofiler.cells AS SELECT * FROM warehouse.cellprofiler.cells;
CREATE TABLE morphem.features   AS SELECT * FROM warehouse.morphem.features;
-- ...repeat for any other table you need, then:
DETACH warehouse;
.shell rm warehouse.duckdb
```

## What is inside this file

`database.jpg` is two files joined end to end. The first part is a JPEG
picture of about 390 KB. The second part is a ZIP archive of about 6 MB. The
archive holds two files:

- `warehouse.duckdb`: the database. DuckDB (a program that keeps tables in
  one file) reads it.
- `README.md`: this file.

A JPEG ends with the two-byte marker `FF D9`, which means that the picture
ends here. A picture viewer stops at that marker. A ZIP archive keeps its
table of contents at the end of the file, so a ZIP tool reads the end first.
Each tool sees only its own part.

Every table is real, measured from CellProfiler's public `ExampleHuman`
tutorial field (CC-0, no personal identifiers).
[Cellpose](https://doi.org/10.1038/s41592-020-01018-x), using its current
default [Cellpose-SAM](https://doi.org/10.1101/2025.04.28.651001) model,
segments the cells from the cell-body channel. CellProfiler-style thresholding
finds nuclei and PH3 foci. `cp_measure` computes the feature columns.

```text
cellprofiler.nuclei    {counts["nuclei"]} rows
cellprofiler.cells     {counts["cells"]} rows
cellprofiler.cytoplasm {counts["cytoplasm"]} rows
cellprofiler.ph3       {counts["ph3"]} rows
images.images          {counts["images_metadata"]} row(s) of image metadata
images.object_crops    {counts["images_object_crops"]} rows, per-cell, per-channel
                       image crops
morphem.features       {counts["morphem_features"]} rows, one real
                       {MORPHEM_EMBEDDING_DIM}-d MorphEm embedding per real cell
morphem.knn_graph      {counts["morphem_knn_graph"]} rows, a real k=6 HNSW
                       k-nearest-neighbor edge list over that embedding
```

`cellprofiler.single_cell` is a view merging Cytoplasm/Cells/Nuclei into one
row per cell. It anchors on Cytoplasm, then LEFT JOINs Cells/Nuclei via
Parent_Cells/Parent_Nuclei. Columns are prefixed
`Cytoplasm_`/`Cells_`/`Nuclei_` to keep the CytoTable-style naming:

```sql
SELECT Cells_AreaShape_Area, Nuclei_Intensity_MeanIntensity_DNA
FROM warehouse.cellprofiler.single_cell
LIMIT 10;
```

Each `images.object_crops` row is a self-describing OME-Arrow-style record:
alongside the crop's own `tile_y`/`tile_x`/`height`/`width`, it also carries
the source field's own `size_c`/`size_z`/`size_y`/`size_x` and `source`/
`source_url`, so no join to `images.images` is required. The pixel data is
stored three ways: `pixel_data_raw` (raw uncompressed array bytes, queryable
with no decoding), `pixel_data_jpegxl` (compact JPEG XL), and
`pixel_data_jpeg` (plain JPEG, for universal compatibility).

Example query:

```sql
SELECT ObjectNumber, channel, size_y, size_x, height, width,
       octet_length(pixel_data_raw) AS raw_bytes,
       octet_length(pixel_data_jpegxl) AS jpegxl_bytes,
       octet_length(pixel_data_jpeg) AS jpeg_bytes
FROM warehouse.images.object_crops
WHERE object_type = 'cells'
LIMIT 10;
```

`morphem.features` holds one real, fixed-size `FLOAT[{MORPHEM_EMBEDDING_DIM}]`
deep-learning embedding per real cell, from the real, pretrained
[{MORPHEM_MODEL}](https://huggingface.co/{MORPHEM_MODEL}) model (a DINO
Vision Transformer, MIT license): each of the DNA/PH3/cell-body channel
crops is embedded independently (384-d each) and concatenated, exactly as
described on the model card. See
`tools/real_example_data/extract_morphem_embeddings.py`.

```sql
SELECT f.ObjectNumber, array_distance(
    f.embedding,
    (SELECT embedding FROM warehouse.morphem.features WHERE ObjectNumber = 1)
) AS distance_from_cell_1
FROM warehouse.morphem.features f
ORDER BY distance_from_cell_1
LIMIT 10;
```

`morphem.features.umap_x`/`umap_y` are a real 2-D
[UMAP](https://umap-learn.readthedocs.io/) projection (McInnes, Healy &
Melville, 2018) of that same embedding, computed with the reference
`umap-learn` implementation using the same Euclidean metric as
`array_distance` and a fixed random seed, so the layout is a faithful,
reproducible visualization of the exact distances the similarity search
runs on.

`morphem.knn_graph` is a real k=6 nearest-neighbor edge list
(`ObjectNumber`, `neighbor_object_number`, `rank`, `distance`), built by
querying a real [HNSW](https://github.com/nmslib/hnswlib) index (Malkov &
Yashunin, 2016) -- the same approximate-nearest-neighbor structure a real
vector index uses for retrieval -- over the same embedding, via the
reference `hnswlib` implementation. `morphem.features.graph_x`/`graph_y`
lay that same graph out with a Fruchterman-Reingold force-directed layout
(`networkx.spring_layout`). `index.html` plots both projections side by
side and draws the real k-NN edges on the graph one, so the two independent
methods' agreement on cluster structure is directly visible.

```sql
SELECT g.ObjectNumber, g.neighbor_object_number, g.rank, g.distance
FROM warehouse.morphem.knn_graph g
WHERE g.ObjectNumber = 1
ORDER BY g.rank;
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
    _draw_example_image_panel(image, quadrants[1])
    _draw_schema_panel(draw, fonts, quadrants[2])
    _draw_guide_panel(draw, fonts, quadrants[3])

    image.save(path, format="JPEG", quality=92, optimize=True)


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


def _draw_example_image_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    padding = 4
    available = min((x1 - x0) - 2 * padding, (y1 - y0) - 2 * padding)
    example_path = ASSETS_DIR / "example_cell_composite.jpg"
    example = (
        Image.open(example_path)
        .convert("RGB")
        .resize((available, available), Image.LANCZOS)
    )
    paste_x = x0 + ((x1 - x0) - example.width) // 2
    paste_y = y0 + ((y1 - y0) - example.height) // 2
    image.paste(example, (paste_x, paste_y))


SCHEMA_TREE_COL_A = [
    ("cellprofiler", ["nuclei", "cells", "cytoplasm", "ph3"]),
]
SCHEMA_TREE_COL_B = [
    ("images", ["images", "object_crops"]),
    ("morphem", ["features", "knn_graph"]),
]


def _draw_schema_panel(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, _, _ = box
    draw.text((x0 + 40, y0 + 24), "Schema", font=fonts["heading"], fill="#20303c")

    font = fonts["mono"]
    line_height = getattr(font, "size", 26) + 16
    root_x, root_y = x0 + 40, y0 + 104
    draw.text((root_x, root_y), "database.jpg", font=font, fill="#20303c")

    col_a_x, col_b_x = root_x, x0 + 320
    stem_x = root_x + 14
    branch_y = root_y + line_height + 16
    tree_top = branch_y + 18
    branch_color = "#3c7d89"
    stem_top_y = root_y + line_height - 8
    draw.line((stem_x, stem_top_y, stem_x, branch_y), fill=branch_color, width=3)
    draw.line((stem_x, branch_y, col_b_x + 14, branch_y), fill=branch_color, width=3)
    draw.line((stem_x, branch_y, stem_x, tree_top), fill=branch_color, width=3)
    draw.line(
        (col_b_x + 14, branch_y, col_b_x + 14, tree_top), fill=branch_color, width=3
    )

    _draw_tree(draw, font, SCHEMA_TREE_COL_A, origin=(col_a_x, tree_top))
    _draw_tree(draw, font, SCHEMA_TREE_COL_B, origin=(col_b_x, tree_top))
    _draw_caption(draw, fonts, box)


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
    top = y1 - 250
    draw.line((x0 + 40, top, x1 - 40, top), fill="#d8ddd2", width=2)
    draw.text(
        (x0 + 40, top + 22), CAPTION_HEADLINE, font=fonts["subtitle"], fill="#20303c"
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
        (x0 + 40, top + 76),
        "\n".join(lines),
        font=font,
        fill="#17202a",
        spacing=10,
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
        "  SET zipfs_split = '!!';\n"
        "  COPY (SELECT content FROM read_blob(\n"
        "    'zip://database.jpg!!warehouse.duckdb'))\n"
        "    TO 'warehouse.duckdb' (FORMAT blob);\n"
        "  ATTACH 'warehouse.duckdb' AS database\n"
        "    (READ_ONLY);\n"
        "  SELECT * FROM database.cellprofiler.cells\n"
        "  LIMIT 10;"
    )
    draw.multiline_text(
        (x0 + 40, y0 + 90),
        guide,
        font=fonts["mono_small"],
        fill="#17202a",
        spacing=16,
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
        "mono_small": load(mono_candidates + candidates, 22),
    }


def _draw_tree(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    tree: list[tuple[str, list[str]]],
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Render a real directory-tree-style listing (like the `tree` command):
    one top-level line per schema, its tables as branches underneath."""

    x, y = origin
    line_height = getattr(font, "size", 26) + 16
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
        archive.write(paths.duckdb, arcname="warehouse.duckdb")
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
    h1 {
      margin: 0 0 4px;
      font-size: 24px;
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
    .plot-intro {
      font-size: 13px;
      margin: 0 0 10px;
    }
    .cluster-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 8px;
    }
    .cluster-grid .panel {
      margin-bottom: 0;
    }
    .cluster-canvas {
      width: 100%;
      height: auto;
      max-width: 100%;
      display: block;
      background: #0c1218;
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
    <h1>A picture is worth a thousand records</h1>
    <p>
      <code>database.jpg</code> opens as a normal picture. It also carries a
      DuckDB database. The picture comes first. After it, the same file stores
      real microscope crops, cell measurements, and model embeddings. An
      embedding is a list of numbers that describes one cell.
    </p>
    <p>
      This is a human-centered file for image-based science. First, you see
      the instructions, schema, and cells. Then an embedded DuckDB database
      lets you query the same file in your browser. The plots show how
      MorphEm, a pretrained model, groups cells that look alike.
    </p>
    <img class="cover" src="database.jpg" alt="Database JPEG cover image">
    <div class="panel">
      <h2>Run your own SQL</h2>
      <p>
        Start with a question. This box runs real SQL against the database.
        Pick an example query, or write your own.
      </p>
      <div class="examples" id="sql-examples"></div>
      <textarea id="sql-editor" spellcheck="false"></textarea>
      <button id="run-sql" type="button" disabled>Run query</button>
      <pre id="sql-status">Waiting for database.jpg to load...</pre>
      <div id="sql-results" aria-live="polite"></div>
    </div>
    <p>
      Each point below is one real cell. Similar cells sit near each other.
      The two maps use different methods, so you can compare them.
    </p>
    <div class="cluster-grid">
      <figure class="panel">
        <h2>UMAP projection</h2>
        <p class="plot-intro">
          UMAP places each cell as a point on a flat map. Similar cells end
          up close together.
        </p>
        <canvas id="umap-canvas" class="cluster-canvas" width="480"
                height="480" role="img" aria-label="UMAP scatter plot of
                real MorphEm embeddings. Click a point to view its nearest
                neighbors."></canvas>
        <figcaption>
          <div class="legend">
            <span>
              <span class="swatch" style="background:#e07a5f"></span>
              search seed
            </span>
            <span>
              <span class="swatch" style="background:#79c2d0"></span>
              nearest neighbors
            </span>
            <span>
              <span class="swatch" style="background:#3c4c58"></span>
              other real cells
            </span>
          </div>
        </figcaption>
      </figure>
      <figure class="panel">
        <h2>HNSW k-NN graph layout</h2>
        <p class="plot-intro">
          HNSW is an index: a structure that quickly finds nearest neighbors.
          This plot draws the cells it connected.
        </p>
        <canvas id="graph-canvas" class="cluster-canvas" width="480"
                height="480" role="img" aria-label="HNSW k-nearest-neighbor
                graph layout of real MorphEm embeddings. Click a point to
                view its nearest neighbors."></canvas>
        <figcaption>
          <div class="legend">
            <span>
              <span class="swatch" style="background:#e07a5f"></span>
              search seed
            </span>
            <span>
              <span class="swatch" style="background:#79c2d0"></span>
              nearest neighbors
            </span>
            <span>
              <span class="swatch" style="background:#3c4c58"></span>
              other real cells
            </span>
            <span>
              <span class="swatch line" style="background:#2a3742"></span>
              HNSW k-NN edge
            </span>
          </div>
        </figcaption>
      </figure>
    </div>
    <p>
      Search for a cell by number, or let the page pick one. The maps will
      highlight that cell and its nearest neighbors.
    </p>
    <div class="search-row">
      <input id="object-number" type="number" min="1" placeholder="Cell #"
             autocomplete="off">
      <button id="search" type="button" disabled>Find similar</button>
      <button id="random" class="secondary" type="button" disabled>Random cell</button>
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
        <img id="seed-image" alt="Search seed cell">
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
      <p>
        The file <code>database.jpg</code> is two files joined end to end.
        The first part is a JPEG picture. The second part is a ZIP archive.
        The archive holds two files.
      </p>
      <ul>
        <li><code>warehouse.duckdb</code>: the database. DuckDB (a program
          that keeps tables in one file) reads it.</li>
        <li><code>README.md</code>: the instructions for the database.</li>
      </ul>
      <h3>How can one file be both?</h3>
      <p>
        A JPEG ends with a two-byte marker, <code>FF D9</code>. The marker
        means that the picture ends here. A picture viewer stops at this
        marker and ignores every byte after it.
      </p>
      <p>
        A ZIP archive keeps its table of contents at the end of the file.
        A ZIP tool reads the end first. The table of contents tells the tool
        where each file starts, so the tool never needs the bytes at the
        front.
      </p>
      <p>
        As a result, each tool sees only its own part. The picture viewer
        sees a picture. The ZIP tool sees an archive. The ZIP data does not
        change the picture. The archive stores the database without
        compression, so the database inside it is a byte-for-byte copy of
        <code>warehouse.duckdb</code>.
      </p>
    </section>
    <section class="text-section">
      <h2>Why this shape?</h2>
      <p>
        We drew inspiration from
        <a href="https://doi.org/10.1145/3749163" target="_blank"
           rel="noopener">F3</a>, a data-file design that stores data,
        metadata, and WebAssembly decoders together. We also drew on
        <a href="https://arxiv.org/abs/2608.07632" target="_blank"
           rel="noopener">JUMP-lite</a>, which focuses on compact,
        reproducible image-based profiling data. This project takes a simpler
        path for image-based science. The code lives on this page. The file
        opens first as an image, so anyone can see what the data is about
        before they run a query.
      </p>
    </section>
    <section class="text-section">
      <h2>How this all works</h2>
      <p>
        The cell crops start with
        <a href="https://doi.org/10.1038/s41592-020-01018-x" target="_blank"
           rel="noopener">Cellpose</a>, a general cell segmentation method.
        Cellpose segments the cell-body channel. CellProfiler-style
        thresholding finds nuclei and PH3 foci. Then
        <a href="https://arxiv.org/abs/2507.01163" target="_blank"
           rel="noopener">cp_measure</a> computes the feature tables.
      </p>
      <p>
        Everything above uses the same real data. Each segmented cell has an
        embedding in <code>morphem.features.embedding</code>. The embedding
        has 1152 numbers from the pretrained
        <a href="https://huggingface.co/CaicedoLab/MorphEm" target="_blank"
           rel="noopener">CaicedoLab/MorphEm</a> model. DuckDB compares two
        embeddings with <code>array_distance</code>. This page uses that same
        function when you search or click a cell.
      </p>
      <p>
        The two plots draw the same embeddings in two ways.
        <a href="https://umap-learn.readthedocs.io/" target="_blank"
           rel="noopener">UMAP</a> (McInnes, Healy &amp; Melville, 2018)
        makes a flat map of the cells. It measures distance the same way that
        <code>array_distance</code> does.
      </p>
      <p>
        The HNSW graph uses a different method. An
        <a href="https://github.com/nmslib/hnswlib" target="_blank"
           rel="noopener">HNSW</a> index (Malkov &amp; Yashunin, 2016) finds
        each cell's 6 nearest neighbors. Then
        <a href="https://networkx.org/" target="_blank"
           rel="noopener">networkx</a> draws the graph. The lines are real
        neighbor links, not a copy of the UMAP map.
      </p>
    </section>
    <section class="text-section">
      <h2>Does this match the source paper?</h2>
      <p>
        The clusters connect to the study that produced this data. Jason
        Moffat, a co-author of that study
        (<a href="https://doi.org/10.1016/j.cell.2006.01.040" target="_blank"
            rel="noopener">Moffat et al. 2006, Cell</a>), provided these
        images. The study used PH3 staining to measure mitotic index: the
        fraction of cells caught while they divide.
      </p>
      <p>
        This warehouse finds
        <strong>__MITOTIC_CELLS__ real PH3-positive (mitotic) cells</strong>
        out of <strong>__TOTAL_CELLS__</strong> in this one field. No one
        told the model which cells were mitotic. Still, those cells cluster
        together. On average,
        <strong>__PURITY_PCT__%</strong> of a PH3-positive cell's nearest
        neighbors, found by the HNSW index, are also PH3-positive. Random
        scattering of mitotic cells predicts only
        <strong>__BASELINE_PCT__%</strong>.
      </p>
      <p>
        MorphEm never trained on this screen. It still finds the same
        mitotic-versus-resting difference, using only image shape and texture.
        This one field cannot recover the study's gene-by-gene results. But
        the cluster pattern gives a real, checkable link to the source data.
      </p>
    </section>
    <section class="text-section">
      <h2>References</h2>
      <p>
        Here is where the data, the models, and the methods on this page
        came from.
      </p>
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
          Moffat, J., Grueneberg, D.A., Yang, X., et al. (2006).
          <a href="https://doi.org/10.1016/j.cell.2006.01.040"
             target="_blank" rel="noopener">A lentiviral RNAi library for
          human and mouse genes applied to an arrayed viral high-content
          screen</a>. Cell, 124(6), 1283 to 1298. Source of the microscope
          images and the mitotic-index screen used above.
        </li>
        <li>
          Carpenter, A.E., Jones, T.R., Lamprecht, M.R., et al. (2006).
          <a href="https://doi.org/10.1186/gb-2006-7-10-r100" target="_blank"
             rel="noopener">CellProfiler: image analysis software for
          identifying and quantifying cell phenotypes</a>. Genome Biology,
          7(10), R100.
        </li>
        <li>
          Stringer, C., Wang, T., Michaelos, M., &amp; Pachitariu, M. (2021).
          <a href="https://doi.org/10.1038/s41592-020-01018-x" target="_blank"
             rel="noopener">Cellpose: a generalist algorithm for cellular
          segmentation</a>. Nature Methods, 18, 100 to 106. Used to segment
          cells from the cell-body channel.
        </li>
        <li>
          Pachitariu, M., &amp; Stringer, C. (2022).
          <a href="https://doi.org/10.1038/s41592-022-01663-4" target="_blank"
             rel="noopener">Cellpose 2.0: how to train your own model</a>.
          Nature Methods, 19, 1634 to 1641.
        </li>
        <li>
          Pachitariu, M., Rariden, M., &amp; Stringer, C. (2025).
          <a href="https://doi.org/10.1101/2025.04.28.651001" target="_blank"
             rel="noopener">Cellpose-SAM: superhuman generalization for
          cellular segmentation</a>. bioRxiv. Cited because the current
          Cellpose package uses Cellpose-SAM as its default model.
        </li>
        <li>
          Munoz, A.F., Treis, T., Kalinin, A.A., Dasgupta, S., Theis, F.,
          Carpenter, A.E., &amp; Singh, S. (2025).
          <a href="https://arxiv.org/abs/2507.01163" target="_blank"
             rel="noopener">cp_measure: API-first feature extraction for
          image-based profiling workflows</a>. arXiv:2507.01163. Used to
          measure the real CellProfiler-style features in this
          database.
        </li>
        <li>
          <a href="https://github.com/cytomining/CytoTable" target="_blank"
             rel="noopener">CytoTable</a> (cytomining project). Source of
          the ExampleHuman test fixtures and the join this database's
          <code>single_cell</code> view follows.
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
          used to compute each cell's embedding.
        </li>
        <li>
          Munoz, A.F., Haslum, J.F., Shen, R., Carpenter, A.E., &amp;
          Singh, S. (2026).
          <a href="https://arxiv.org/abs/2608.07632" target="_blank"
             rel="noopener">JUMP-lite: Compact, reproducible benchmarking of
          cell representations</a>. arXiv:2608.07632. Benchmarks
          CellProfiler and MorphEm using lossy JPEG XL compression, the
          same codec this database uses for <code>pixel_data_jpegxl</code>.
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
        <li>
          Hagberg, A.A., Schult, D.A., &amp; Swart, P.J. (2008).
          <a href="https://networkx.org/" target="_blank"
             rel="noopener">Exploring network structure, dynamics, and
          function using NetworkX</a>. Proceedings of the 7th Python in
          Science Conference (SciPy 2008), 11 to 15.
        </li>
      </ul>
    </section>
  </main>
  <script type="module">
    import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.33.1-dev57.0/+esm";
    import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.2/+esm";

    const statusEl = document.querySelector("#status");
    const searchButton = document.querySelector("#search");
    const randomButton = document.querySelector("#random");
    const objectNumberInput = document.querySelector("#object-number");
    const seedFigure = document.querySelector("#seed-figure");
    const seedImage = document.querySelector("#seed-image");
    const seedCaption = document.querySelector("#seed-caption");
    const resultsEl = document.querySelector("#results");
    const umapCanvas = document.querySelector("#umap-canvas");
    const graphCanvas = document.querySelector("#graph-canvas");
    const umapCtx = umapCanvas.getContext("2d");
    const graphCtx = graphCanvas.getContext("2d");
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
    let umapPoints = [];
    let graphPoints = [];
    let graphEdges = [];
    let pointsByObjectNumber = new Map();
    const DEFAULT_OBJECT_NUMBER = "269";

    const fetchJpegBytes = async () => {
      setStatus("Fetching database.jpg...");
      const response = await fetch("database.jpg", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`database.jpg returned ${response.status}`);
      }
      return response.arrayBuffer();
    };

    const openWarehouse = async (jpegBytes) => {
      setStatus("Extracting warehouse.duckdb from JPEG ZIP payload...");
      const zip = await JSZip.loadAsync(jpegBytes);
      const databaseMember = zip.file("warehouse.duckdb");
      if (!databaseMember) {
        throw new Error("warehouse.duckdb was not found in the JPEG ZIP payload");
      }
      const databaseBytes = await databaseMember.async("uint8array");

      setStatus("Starting DuckDB-Wasm...");
      const bundles = duckdb.getJsDelivrBundles();
      const bundle = await duckdb.selectBundle(bundles);
      const worker = await duckdb.createWorker(bundle.mainWorker);
      const database = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
      await database.instantiate(bundle.mainModule, bundle.pthreadWorker);
      await database.registerFileBuffer("warehouse.duckdb", databaseBytes);

      const conn = await database.connect();
      await conn.query("ATTACH 'warehouse.duckdb' AS warehouse (READ_ONLY)");
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

    const projectPoints = (points, xKey, yKey, canvas) => {
      const xs = points.map((point) => point[xKey]);
      const ys = points.map((point) => point[yKey]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = maxX - minX || 1;
      const spanY = maxY - minY || 1;
      const pad = 24;
      const w = canvas.width - 2 * pad;
      const h = canvas.height - 2 * pad;
      return points.map((point) => ({
        objectNumber: point.objectNumber,
        // flip y: canvas y grows downward, plot axes conventionally point up.
        canvasX: pad + ((point[xKey] - minX) / spanX) * w,
        canvasY: pad + (1 - (point[yKey] - minY) / spanY) * h,
      }));
    };

    const loadClusterData = async () => {
      // one real row per real cell, cheap to load in full,
      // including its real cellbody JPEG for instant hover previews.
      const pointTable = await connection.query(`
        SELECT
            f.ObjectNumber, f.umap_x, f.umap_y, f.graph_x, f.graph_y,
            oc.pixel_data_jpeg
        FROM warehouse.morphem.features f
        JOIN warehouse.images.object_crops oc
            ON oc.ObjectNumber = f.ObjectNumber
            AND oc.object_type = 'cells' AND oc.channel = 'cellbody'
      `);
      pointsByObjectNumber = new Map();
      for (const row of rowsOf(pointTable)) {
        const objectNumber = String(cellValue(row.ObjectNumber));
        pointsByObjectNumber.set(objectNumber, {
          objectNumber,
          umapX: Number(row.umap_x),
          umapY: Number(row.umap_y),
          graphX: Number(row.graph_x),
          graphY: Number(row.graph_y),
          imageDataUrl: jpegToDataUrl(row.pixel_data_jpeg),
        });
      }
      const allPoints = Array.from(pointsByObjectNumber.values());
      umapPoints = projectPoints(allPoints, "umapX", "umapY", umapCanvas);
      graphPoints = projectPoints(allPoints, "graphX", "graphY", graphCanvas);

      // the real k-NN edges a real HNSW index found for each real cell.
      const edgeTable = await connection.query(`
        SELECT ObjectNumber, neighbor_object_number
        FROM warehouse.morphem.knn_graph
      `);
      graphEdges = rowsOf(edgeTable).map((row) => ({
        a: String(cellValue(row.ObjectNumber)),
        b: String(cellValue(row.neighbor_object_number)),
      }));
    };

    const drawPointDots = (ctx, points, seedObjectNumber, neighborObjectNumbers) => {
      const neighborSet = new Set(neighborObjectNumbers ?? []);
      for (const point of points) {
        const isSeed = point.objectNumber === seedObjectNumber;
        const isNeighbor = neighborSet.has(point.objectNumber);
        ctx.beginPath();
        const radius = isSeed ? 7 : isNeighbor ? 5 : 3.5;
        ctx.arc(point.canvasX, point.canvasY, radius, 0, Math.PI * 2);
        ctx.fillStyle = isSeed ? "#e07a5f" : isNeighbor ? "#79c2d0" : "#3c4c58";
        ctx.fill();
        if (isSeed) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#f6f7f2";
          ctx.stroke();
        }
      }
    };

    const drawUmap = (seedObjectNumber, neighborObjectNumbers) => {
      umapCtx.clearRect(0, 0, umapCanvas.width, umapCanvas.height);
      drawPointDots(umapCtx, umapPoints, seedObjectNumber, neighborObjectNumbers);
    };

    const drawGraph = (seedObjectNumber, neighborObjectNumbers) => {
      graphCtx.clearRect(0, 0, graphCanvas.width, graphCanvas.height);
      const byId = new Map(graphPoints.map((point) => [point.objectNumber, point]));
      graphCtx.strokeStyle = "#2a3742";
      graphCtx.lineWidth = 1;
      for (const edge of graphEdges) {
        const a = byId.get(edge.a);
        const b = byId.get(edge.b);
        if (!a || !b) {
          continue;
        }
        graphCtx.beginPath();
        graphCtx.moveTo(a.canvasX, a.canvasY);
        graphCtx.lineTo(b.canvasX, b.canvasY);
        graphCtx.stroke();
      }
      drawPointDots(graphCtx, graphPoints, seedObjectNumber, neighborObjectNumbers);
    };

    const redrawClusters = (seedObjectNumber, neighborObjectNumbers) => {
      const neighbors = neighborObjectNumbers ?? [];
      drawUmap(seedObjectNumber, neighbors);
      drawGraph(seedObjectNumber, neighbors);
    };

    const canvasCoords = (canvas, event) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
        scale: (scaleX + scaleY) / 2,
      };
    };

    const nearestOf = (points, x, y) => {
      let nearest = null;
      let nearestDistSq = Infinity;
      for (const point of points) {
        const dx = point.canvasX - x;
        const dy = point.canvasY - y;
        const distSq = dx * dx + dy * dy;
        if (distSq < nearestDistSq) {
          nearestDistSq = distSq;
          nearest = point;
        }
      }
      return nearest;
    };

    const attachClusterInteraction = (canvas, pointsGetter) => {
      canvas.addEventListener("click", (event) => {
        const points = pointsGetter();
        if (!points.length) {
          return;
        }
        const { x, y } = canvasCoords(canvas, event);
        const nearest = nearestOf(points, x, y);
        if (nearest) {
          runSearch(nearest.objectNumber);
        }
      });
      canvas.addEventListener("mousemove", (event) => {
        const points = pointsGetter();
        if (!points.length) {
          return;
        }
        const { x, y, scale } = canvasCoords(canvas, event);
        const nearest = nearestOf(points, x, y);
        const distance = nearest
          ? Math.hypot(nearest.canvasX - x, nearest.canvasY - y)
          : Infinity;
        if (!nearest || distance > 14 * scale) {
          tooltip.hidden = true;
          canvas.style.cursor = "default";
          return;
        }
        canvas.style.cursor = "pointer";
        const point = pointsByObjectNumber.get(nearest.objectNumber);
        tooltipImage.src = point.imageDataUrl;
        tooltipLabel.textContent = `cell ${nearest.objectNumber}`;
        tooltip.style.left = `${event.clientX + 14}px`;
        tooltip.style.top = `${event.clientY + 14}px`;
        tooltip.hidden = false;
      });
      canvas.addEventListener("mouseleave", () => {
        tooltip.hidden = true;
      });
    };

    attachClusterInteraction(umapCanvas, () => umapPoints);
    attachClusterInteraction(graphCanvas, () => graphPoints);

    const renderSeed = (row) => {
      seedFigure.hidden = false;
      seedImage.src = jpegToDataUrl(row.pixel_data_jpeg);
      seedCaption.textContent = `cell ${cellValue(row.ObjectNumber)} (search seed)`;
    };

    const renderResults = (rows) => {
      resultsEl.replaceChildren();
      for (const row of rows) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "result";
        const img = document.createElement("img");
        img.src = jpegToDataUrl(row.pixel_data_jpeg);
        img.alt = `cell ${cellValue(row.ObjectNumber)}`;
        const meta = document.createElement("div");
        meta.className = "meta";
        const idLine = document.createElement("span");
        idLine.textContent = `cell ${cellValue(row.ObjectNumber)}`;
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
        setStatus("Enter a whole cell number.");
        return;
      }
      searchButton.disabled = true;
      randomButton.disabled = true;
      try {
        setStatus(`Finding cells similar to cell ${objectNumber}...`);
        const seedTable = await connection.query(`
          SELECT oc.ObjectNumber, oc.pixel_data_jpeg
          FROM warehouse.images.object_crops oc
          WHERE oc.object_type = 'cells' AND oc.channel = 'cellbody'
              AND oc.ObjectNumber = ${objectNumber}
          LIMIT 1
        `);
        const seedRows = rowsOf(seedTable);
        if (seedRows.length === 0) {
          setStatus(`No cell found with number ${objectNumber}.`);
          resultsEl.replaceChildren();
          seedFigure.hidden = true;
          return;
        }
        renderSeed(seedRows[0]);

        const neighborTable = await connection.query(`
          SELECT
              f.ObjectNumber,
              oc.pixel_data_jpeg,
              array_distance(
                  f.embedding,
                  (
                      SELECT embedding FROM warehouse.morphem.features
                      WHERE ObjectNumber = ${objectNumber}
                  )
              ) AS dist
          FROM warehouse.morphem.features f
          JOIN warehouse.images.object_crops oc
              ON oc.ObjectNumber = f.ObjectNumber
              AND oc.object_type = 'cells' AND oc.channel = 'cellbody'
          WHERE f.ObjectNumber != ${objectNumber}
          ORDER BY dist
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
        );
        setStatus(`Showing cells similar to cell ${objectNumber} (real embeddings).`);
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
        FROM warehouse.morphem.features
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
        label: "array_distance",
        sql: `SELECT ObjectNumber, array_distance(
    embedding,
    (SELECT embedding FROM warehouse.morphem.features WHERE ObjectNumber = 1)
) AS distance
FROM warehouse.morphem.features
ORDER BY distance
LIMIT 10;`,
      },
      {
        label: "knn_graph",
        sql: `SELECT g.ObjectNumber, g.neighbor_object_number, g.rank, g.distance
FROM warehouse.morphem.knn_graph g
WHERE g.ObjectNumber = 1
ORDER BY g.rank;`,
      },
      {
        label: "single_cell join",
        sql: `SELECT Cells_AreaShape_Area, Nuclei_Intensity_MeanIntensity_DNA
FROM warehouse.cellprofiler.single_cell
LIMIT 10;`,
      },
      {
        label: "object_crops",
        sql: `SELECT ObjectNumber, channel, size_y, size_x, height, width,
       octet_length(pixel_data_raw) AS raw_bytes,
       octet_length(pixel_data_jpegxl) AS jpegxl_bytes,
       octet_length(pixel_data_jpeg) AS jpeg_bytes
FROM warehouse.images.object_crops
WHERE object_type = 'cells'
LIMIT 10;`,
      },
      {
        label: "Row counts",
        sql: `SELECT 'morphem.features' AS table_name, COUNT(*) AS rows
FROM warehouse.morphem.features
UNION ALL
SELECT 'morphem.knn_graph', COUNT(*) FROM warehouse.morphem.knn_graph
UNION ALL
SELECT 'images.object_crops', COUNT(*) FROM warehouse.images.object_crops;`,
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
        connection = await openWarehouse(await loadBytes());
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
""".replace("__MITOTIC_CELLS__", str(counts["mitotic_cells"]))
        .replace("__TOTAL_CELLS__", str(counts["cells"]))
        .replace("__PURITY_PCT__", str(counts["mitotic_knn_purity_pct"]))
        .replace("__BASELINE_PCT__", str(counts["mitotic_baseline_pct"]))
    )
    path.write_text(html, encoding="utf-8")
