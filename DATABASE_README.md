# JPEG Database

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
bbbc039.images          200 rows, one field of view each
bbbc039.objects         19565 rows, one manual nucleus each
bbbc039.annotations     view over the same manual objects
bbbc039.segmentations   view with one row per annotated mask
images.images           200 rows, image metadata (no pixels)
images.object_crops     19565 rows, one JPEG crop per nucleus
features.cp_measure     19565 cp_measure PCA vectors (16-d)
features.morphem        19565 MorphEm embedding vectors
features.fused          19565 fused feature vectors
features.knn_graph      352170 neighbor edges (k=6)
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
The [cellpose](https://github.com/MouseLand/cellpose) library (version 4.2.1.1, model
`cpsam_v2`) segmented the same DNA fields, and a manual nucleus counts
as captured (`cellpose_captured = true`) when a Cellpose mask overlapped it
with IoU >= 0.5 under one-to-one matching. Cellpose captured
90.4% of the 19,565 manually
annotated nuclei. `cellpose_iou` holds the matched pair's IoU for captured
nuclei and the best IoU reached with any mask otherwise. The Cellpose masks
themselves are not stored; re-run `tools/bbbc039_data/match_cellpose.py` to
reproduce them.
IoU means intersection over union: the shared area divided by the total area
covered by either mask. It is also called the
[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index). An IoU of 1
means perfect overlap, and 0 means no overlap.

The page overlays the 1,878 manual nuclei Cellpose missed on the PCA, UMAP,
and t-SNE maps. Those maps were computed from manual masks and DNA crops; the
Cellpose verdict was not part of the feature vector. If missed nuclei cluster,
that suggests they share measured shape, intensity, or learned visual
neighborhoods. If they scatter, the miss may depend on image context, touching
cells, or segmentation behavior that these nucleus-level features do not
capture. cp_measure is easiest to interpret through explicit size, shape,
brightness, texture, and radial intensity columns; MorphEm can group subtler
crop appearance but is harder to explain; Fused combines both, but you still
need to query the source columns and crops to decide which side explains a
miss.

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
[BBBC039](https://bbbc.broadinstitute.org/BBBC039). The only pixels in the database are the JPEG crops.

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
pretrained [CaicedoLab/MorphEm](https://huggingface.co/CaicedoLab/MorphEm) model on
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
