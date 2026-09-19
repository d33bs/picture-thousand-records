# JPEG Warehouse Prototype Spec

## Goal

Build a **single standalone `.jpg` file** that is both:

1. A normal JPEG that opens in Chrome and can be hosted on GitHub Pages.
2. A self-contained analytical package containing a DuckDB database.
3. Queryable with DuckDB through a ZIP filesystem path.
4. Usable locally from a notebook for CytoDataFrame-based exploration of CellProfiler and image data.
5. Self-documenting: the visible JPEG contains the system design, contents, and instructions for opening/querying the embedded database.

Working name: **JPEG Warehouse**.

---

# Primary architecture

Use a **JPEG + ZIP polyglot**, with a DuckDB database stored inside the ZIP.

```text
warehouse.jpg
┌──────────────────────────────────────────────┐
│ JPEG                                         │
│                                              │
│ visible system diagram                      │
│ dataset summary                             │
│ SQL / access instructions                   │
│                                              │
│ JPEG EOI marker                             │
├──────────────────────────────────────────────┤
│ ZIP                                          │
│                                              │
│ warehouse.duckdb                            │
│ README.md                                   │
│ manifest.json                               │
│                                              │
├──────────────────────────────────────────────┤
│ ZIP central directory                       │
│ ZIP EOCD                                    │
└──────────────────────────────────────────────┘
```

The JPEG is the **human-facing interface**.

The embedded DuckDB database is the **machine-facing interface**.

The ZIP layer makes the embedded database addressable with an existing filesystem abstraction rather than inventing a new JPEG-aware DuckDB file format.

---

# Expected access pattern

The target DuckDB interface is:

```sql
INSTALL zipfs FROM community;
LOAD zipfs;

SET zipfs_split = '!!';

ATTACH 'zip://warehouse.jpg!!warehouse.duckdb'
    AS warehouse
    (READ_ONLY);
```

Then:

```sql
SELECT *
FROM warehouse.main.samples
LIMIT 10;
```

The prototype should explicitly test this path with the current `zipfs` extension.

If direct `ATTACH` through `zipfs` is not supported by a given DuckDB build, the fallback is:

1. Read `warehouse.duckdb` through the ZIP filesystem.
2. Materialize/register the database locally.
3. Attach the resulting local database.

The project should treat **direct `zipfs` + `ATTACH` as the preferred path**, but keep the fallback isolated behind a small helper.

---

# Why embed DuckDB instead of separate Parquet files

The first version should package one complete `warehouse.duckdb` database.

Advantages:

- one logical database,
- schemas, views, and relationships live with the data,
- SQL examples can reference stable table names,
- CellProfiler and image tables can be joined directly,
- DuckDB types such as `BLOB`, arrays, structs, and nested values can be preserved,
- notebook users can `ATTACH` one database rather than register many files,
- the JPEG remains a single portable artifact.

Parquet may still be used internally when building the database, but it is not the primary distribution format.

---

# Example data

All human data must be **synthetic**.

The demo represents high-content microscopy experiments using human-derived cell samples.

Suggested scale:

```text
100 synthetic people
500 samples
2,500 images
~500,000 segmented cells
```

This is large enough to make joins and aggregation interesting while staying practical for a local demo.

---

# DuckDB schemas

Use logical schemas to separate data domains.

```text
warehouse.duckdb

main
├── people
└── samples

cellprofiler
├── images
└── cells

ome_arrow
├── images
└── tiles

media
└── jpeg_images
```

---

# Synthetic human data

## `main.people`

One row per synthetic donor.

```text
person_id          VARCHAR
age_years          INTEGER
sex                VARCHAR
ancestry_group     VARCHAR
cohort              VARCHAR
```

Do not include names, dates of birth, addresses, medical record numbers, or real identifiers.

## `main.samples`

One row per biological sample.

```text
sample_id      VARCHAR
person_id      VARCHAR
cell_line      VARCHAR
plate_id       VARCHAR
well           VARCHAR
treatment      VARCHAR
dose_um        DOUBLE
```

Relationship:

```text
people.person_id
        ↓
samples.person_id
```

---

# CellProfiler data

## `cellprofiler.images`

One row per acquired image/site.

```text
image_id              VARCHAR
sample_id             VARCHAR
plate_id              VARCHAR
well                  VARCHAR
site                  INTEGER
width                 INTEGER
height                INTEGER
channel_count         INTEGER
cell_count            INTEGER
focus_score           DOUBLE
illumination_score    DOUBLE
```

## `cellprofiler.cells`

One row per segmented cell.

```text
object_id                   BIGINT
image_id                    VARCHAR
cell_number                 INTEGER
center_x                    DOUBLE
center_y                    DOUBLE
area_shape_area             DOUBLE
area_shape_eccentricity     DOUBLE
intensity_mean_dna          DOUBLE
intensity_mean_rna          DOUBLE
intensity_mean_agp          DOUBLE
intensity_mean_mito         DOUBLE
intensity_mean_er           DOUBLE
texture_entropy_dna         DOUBLE
neighbors_number            INTEGER
```

This table should be large enough to demonstrate group-by aggregation, joins, filtering, morphology queries, and per-image/per-treatment summaries.

---

# OME-Arrow-style images

The demo should include an OME-Arrow-compatible or OME-Arrow-inspired representation that can be passed from DuckDB into Python without relying on image file paths.

## `ome_arrow.images`

One row per logical image.

```text
image_id         VARCHAR
sample_id        VARCHAR
size_c           INTEGER
size_z           INTEGER
size_y           INTEGER
size_x           INTEGER
dtype            VARCHAR
physical_x_um    DOUBLE
physical_y_um    DOUBLE
physical_z_um    DOUBLE
tile_y           INTEGER
tile_x           INTEGER
codec            VARCHAR
```

## `ome_arrow.tiles`

One row per image tile.

```text
image_id       VARCHAR
channel        INTEGER
z              INTEGER
tile_y         INTEGER
tile_x         INTEGER
height         INTEGER
width          INTEGER
dtype          VARCHAR
codec          VARCHAR
pixel_data     BLOB
```

Suggested initial `codec` values:

```text
raw
jpeg-xl
```

The database should allow DuckDB to query tile metadata without decoding image bytes.

Example:

```sql
SELECT
    image_id,
    channel,
    COUNT(*) AS tile_count,
    SUM(octet_length(pixel_data)) AS encoded_bytes
FROM warehouse.ome_arrow.tiles
GROUP BY image_id, channel;
```

The notebook layer is responsible for decoding and displaying the selected image data.

---

# JPEG images stored directly in DuckDB

## `media.jpeg_images`

This table demonstrates ordinary JPEG images as first-class database values.

```text
jpeg_id         VARCHAR
image_id        VARCHAR
purpose         VARCHAR
width           INTEGER
height          INTEGER
mime_type       VARCHAR
jpeg_bytes      BLOB
```

Suggested `purpose` values:

```text
thumbnail
crop
segmentation_preview
qc_preview
```

Example SQL:

```sql
SELECT
    image_id,
    jpeg_bytes
FROM warehouse.media.jpeg_images
WHERE purpose = 'thumbnail'
LIMIT 10;
```

A Python notebook can turn each `jpeg_bytes` value into a displayable image without requiring a separate file.

---

# Relationships

```text
main.people
    │ person_id
    ↓
main.samples
    │ sample_id
    ├─────────────────────────┐
    ↓                         ↓
cellprofiler.images      ome_arrow.images
    │ image_id                 │ image_id
    ↓                          ↓
cellprofiler.cells       ome_arrow.tiles
    │
    └──────────────────────────────→ media.jpeg_images
                                   image_id
```

These relationships should also appear visually in the JPEG cover image.

---

# JPEG cover image

The visible JPEG is part of the format design.

It should contain:

## Title

```text
JPEG Warehouse
Synthetic Cell Imaging Demo
```

## Dataset summary

```text
100 synthetic people
500 biological samples
2,500 microscopy images
~500,000 segmented cells
CellProfiler features
OME-Arrow image payloads
JPEG previews
```

## System diagram

```text
┌──────────┐
│  People  │
└────┬─────┘
     ↓
┌──────────┐
│ Samples  │
└────┬─────┘
     ↓
┌──────────────────────┐
│ Microscopy Images    │
├──────────────────────┤
│ CellProfiler         │
│ OME-Arrow            │
│ JPEG previews        │
└──────────────────────┘
```

## Access instructions

The image should explicitly tell a technical user that the JPEG contains a database.

Example:

```text
This JPEG contains an embedded DuckDB database.

DuckDB:

INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';

ATTACH 'zip://warehouse.jpg!!warehouse.duckdb'
    AS warehouse
    (READ_ONLY);

SELECT *
FROM warehouse.main.samples
LIMIT 10;
```

Also show:

```text
Local notebook demo:
notebooks/cytodataframe_demo.ipynb
```

The goal is that the JPEG itself explains what it contains and how to open it.

---

# ZIP contents

The embedded ZIP should contain:

```text
warehouse.duckdb
README.md
manifest.json
```

Optional later additions:

```text
notebooks/cytodataframe_demo.ipynb
queries/example_queries.sql
LICENSE
```

For the first prototype, the notebook may live in the repository rather than inside the JPEG if that makes development easier.

---

# `manifest.json`

Example:

```json
{
  "format": "jpeg-warehouse",
  "version": "0.1",
  "title": "Synthetic Cell Imaging Warehouse",
  "database": "warehouse.duckdb",
  "database_engine": "duckdb",
  "access": {
    "filesystem": "zipfs",
    "member": "warehouse.duckdb"
  },
  "schemas": [
    "main",
    "cellprofiler",
    "ome_arrow",
    "media"
  ]
}
```

The manifest is descriptive. DuckDB itself remains the source of truth for tables and schemas.

---

# Building the artifact

## Step 1: create `warehouse.duckdb`

Build the database normally and verify it independently before embedding it.

## Step 2: create the JPEG cover

Generate a static JPEG containing the system diagram, dataset summary, SQL instructions, and project/version information.

## Step 3: append the ZIP

Create a ZIP whose main member is `warehouse.duckdb`. Prefer `ZIP_STORED` for the DuckDB database unless testing shows another choice works better.

Conceptually:

```python
import zipfile

with zipfile.ZipFile(
    "warehouse.jpg",
    mode="a",
    compression=zipfile.ZIP_STORED,
) as z:
    z.write("warehouse.duckdb")
    z.write("manifest.json")
    z.write("README.md")
```

The resulting artifact remains one file: `warehouse.jpg`.

---

# Validation

## JPEG view

Open in Chrome, Firefox, and the operating-system image viewer.

Expected result: the system diagram renders normally.

## ZIP view

```bash
unzip -l warehouse.jpg
```

Expected members:

```text
warehouse.duckdb
manifest.json
README.md
```

## DuckDB view

Test:

```sql
INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';

ATTACH 'zip://warehouse.jpg!!warehouse.duckdb'
    AS warehouse
    (READ_ONLY);

SHOW ALL TABLES;
```

If this exact path does not work with current `zipfs`, document the limitation and implement the smallest possible extraction/materialization fallback.

---

# Local CytoDataFrame notebook

Create:

```text
notebooks/cytodataframe_demo.ipynb
```

The notebook demonstrates the artifact from a local scientific Python workflow.

## Notebook flow

### 1. Open the JPEG warehouse

Attach the embedded database.

Preferred:

```python
import duckdb

con = duckdb.connect()

con.execute("""
INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';
""")

con.execute("""
ATTACH 'zip://../warehouse.jpg!!warehouse.duckdb'
AS warehouse
(READ_ONLY)
""")
```

Fallback if `ATTACH` through `zipfs` is unsupported:

```text
extract/materialize warehouse.duckdb
↓
ATTACH local database
```

The fallback should be hidden in a helper so the rest of the notebook is unchanged.

### 2. Query CellProfiler profiles

```python
profiles = con.sql("""
SELECT
    c.*,
    i.sample_id,
    s.person_id,
    s.treatment
FROM warehouse.cellprofiler.cells c
JOIN warehouse.cellprofiler.images i
    USING (image_id)
JOIN warehouse.main.samples s
    USING (sample_id)
LIMIT 1000
""").df()
```

### 3. Create a CytoDataFrame

Conceptually:

```python
from cytodataframe import CytoDataFrame

cdf = CytoDataFrame(profiles)
cdf
```

Use the actual current CytoDataFrame constructor/API in the implementation.

### 4. Display JPEG image BLOBs

Query thumbnail bytes:

```python
result = con.sql("""
SELECT
    c.object_id,
    c.image_id,
    c.area_shape_area,
    j.jpeg_bytes
FROM warehouse.cellprofiler.cells c
JOIN warehouse.media.jpeg_images j
    USING (image_id)
WHERE j.purpose = 'thumbnail'
LIMIT 25
""")
```

The notebook should convert `jpeg_bytes` into an image representation CytoDataFrame can render inline.

Target experience:

| object_id | image_id | area | image |
|---:|---|---:|---|
| 1 | IMG001 | 412 | thumbnail |
| 2 | IMG001 | 377 | thumbnail |

### 5. Display OME-Arrow image data

Query OME-Arrow metadata plus selected image payloads:

```sql
SELECT
    i.image_id,
    i.size_c,
    i.size_y,
    i.size_x,
    t.channel,
    t.tile_y,
    t.tile_x,
    t.codec,
    t.pixel_data
FROM warehouse.ome_arrow.images i
JOIN warehouse.ome_arrow.tiles t
    USING (image_id)
WHERE i.image_id = 'IMG000042';
```

Python reconstructs the image representation from the returned Arrow/DuckDB values.

Target flow:

```text
DuckDB result
    ↓
Arrow / Python values
    ↓
OME-Arrow reconstruction
    ↓
CytoDataFrame image-aware representation
    ↓
inline local display
```

The goal is to prove that image bytes can originate from the embedded DuckDB database rather than from external image paths.

---

# CytoDataFrame demo goals

The notebook should prove four things:

1. CytoDataFrame can display feature rows queried from the embedded DuckDB database.
2. JPEG `BLOB`s returned by DuckDB can be displayed inline.
3. OME-Arrow image bytes returned by DuckDB can be reconstructed and viewed locally.
4. CellProfiler features, biological metadata, and image representations can stay linked by `image_id` and related identifiers.

This makes CytoDataFrame the local scientific exploration layer while DuckDB remains the storage and query layer.

---

# Example queries

## Human metadata + CellProfiler

```sql
SELECT
    p.age_years,
    s.treatment,
    AVG(c.area_shape_area) AS mean_cell_area,
    AVG(c.intensity_mean_mito) AS mean_mito_intensity
FROM warehouse.main.people p
JOIN warehouse.main.samples s
    USING (person_id)
JOIN warehouse.cellprofiler.images i
    USING (sample_id)
JOIN warehouse.cellprofiler.cells c
    USING (image_id)
GROUP BY
    p.age_years,
    s.treatment;
```

## Morphology outliers

```sql
SELECT
    image_id,
    AVG(area_shape_eccentricity) AS mean_eccentricity,
    COUNT(*) AS cells
FROM warehouse.cellprofiler.cells
GROUP BY image_id
HAVING COUNT(*) > 100
ORDER BY mean_eccentricity DESC
LIMIT 20;
```

## Image payload sizes

```sql
SELECT
    image_id,
    codec,
    COUNT(*) AS tiles,
    SUM(octet_length(pixel_data)) AS encoded_bytes
FROM warehouse.ome_arrow.tiles
GROUP BY image_id, codec
ORDER BY encoded_bytes DESC;
```

## JPEG previews as data

```sql
SELECT
    image_id,
    purpose,
    width,
    height,
    octet_length(jpeg_bytes) AS bytes
FROM warehouse.media.jpeg_images
ORDER BY bytes DESC;
```

---

# Browser / web target

The same file should be hostable as:

```text
https://example.github.io/jpeg-warehouse/warehouse.jpg
```

Chrome should render the JPEG cover normally.

A future DuckDB-Wasm demo can investigate whether `zipfs` can access the embedded database directly from the remote JPEG.

That is a later milestone. The **first success criterion is local DuckDB + local notebook access**.

---

# Milestones

## V0 — JPEG + ZIP + DuckDB

Build `warehouse.jpg` with an embedded ZIP containing `warehouse.duckdb`.

Verify:

- Chrome renders JPEG.
- `unzip` lists the database.
- DuckDB can reach the embedded database through `zipfs` or a minimal fallback.

## V1 — scientific relational data

Add:

```text
main.people
main.samples
cellprofiler.images
cellprofiler.cells
```

Demonstrate joins and aggregations.

## V2 — JPEG BLOBs

Add:

```text
media.jpeg_images
```

Demonstrate:

```text
DuckDB query
→ BLOB
→ Python
→ inline image
```

## V3 — OME-Arrow images

Add:

```text
ome_arrow.images
ome_arrow.tiles
```

Demonstrate image reconstruction from DuckDB results.

## V4 — CytoDataFrame notebook

Build:

```text
notebooks/cytodataframe_demo.ipynb
```

Demonstrate:

```text
JPEG warehouse
→ DuckDB
→ CellProfiler rows
→ image BLOBs / OME-Arrow
→ CytoDataFrame
```

## V5 — self-documenting cover

Finalize the JPEG cover so it includes schema, architecture, dataset summary, DuckDB commands, and local notebook instructions.

The file should explain itself when opened as an image.

## V6 — DuckDB-Wasm / GitHub Pages

Host `warehouse.jpg` on GitHub Pages and investigate:

```text
DuckDB-Wasm
→ zipfs
→ embedded warehouse.duckdb
```

If direct remote attachment is not practical, keep the same file format and add the smallest browser-side adapter necessary.

---

# Initial implementation recommendation

Start with this exact artifact:

```text
warehouse.jpg

JPEG:
    system architecture
    schema diagram
    data summary
    DuckDB access instructions

ZIP:
    warehouse.duckdb
    manifest.json
    README.md
```

Inside `warehouse.duckdb`:

```text
main.people
main.samples
cellprofiler.images
cellprofiler.cells
ome_arrow.images
ome_arrow.tiles
media.jpeg_images
```

And build one local notebook:

```text
notebooks/cytodataframe_demo.ipynb
```

The core demonstration is:

> **Open the file as a JPEG to understand the dataset. Attach the same file through DuckDB to query it. Open the query results in CytoDataFrame to explore the associated microscopy images locally.**
