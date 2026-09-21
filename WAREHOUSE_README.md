# JPEG Warehouse

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
cellprofiler.nuclei    274 rows
cellprofiler.cells     303 rows
cellprofiler.cytoplasm 303 rows
cellprofiler.ph3       28 rows
images.images          1 row(s) of image metadata
images.object_crops    909 rows, per-cell, per-channel
                       image crops
morphem.features       303 rows, one real
                       1152-d MorphEm embedding per real cell
morphem.knn_graph      1818 rows, a real k=6 HNSW
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

`morphem.features` holds one real, fixed-size `FLOAT[1152]`
deep-learning embedding per real cell, from the real, pretrained
[CaicedoLab/MorphEm](https://huggingface.co/CaicedoLab/MorphEm) model (a DINO
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
