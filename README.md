# picture-thousand-records

`database.jpg` is a normal JPEG image. It also contains a DuckDB database in
an appended ZIP archive.

The project uses the full
[BBBC039](https://bbbc.broadinstitute.org/BBBC039) benchmark. BBBC039 has 200
DNA-channel microscopy fields of human U2OS cells and manual nucleus masks.
The database stores one JPEG crop per nucleus, the measurements, the embeddings,
and the nearest-neighbor indexes together. It does not store the original
images or masks (see "What The Database Does Not Store").

## What Is Inside

```text
database.jpg
  JPEG image
    representative BBBC039 field of view
  ZIP archive
    database.duckdb
    README.md
```

The DuckDB database has three schemas:

```text
bbbc039.images          200 fields of view
bbbc039.objects         19,565 manual nucleus objects
bbbc039.annotations     view over the manual objects
bbbc039.segmentations   view with one row per mask
images.images           200 rows of image metadata (no pixels)
images.object_crops     one JPEG crop per nucleus
features.cp_measure     16-d PCA of cp_measure measurements
features.morphem        MorphEm DNA-crop embeddings
features.fused          cp_measure plus MorphEm vectors
features.knn_graph      6 nearest neighbors per nucleus, per feature space
```

The page compares three feature spaces:

- `cp_measure`: a 16-dimension PCA reduction of the CellProfiler measurements
  that [cp_measure](https://github.com/afermg/cp_measure) computed for each
  nucleus. See the next section.
- `MorphEm`: learned image features from `CaicedoLab/MorphEm`.
- `Fused`: both feature blocks joined after dimension reduction.

## What The Database Does Not Store

To keep the file smaller, the database holds no original TIFF images, no
manual mask PNGs, no 16-bit raw crops, and no JPEG XL crops. The only image
bytes are the 8-bit JPEG crops in `images.object_crops.pixel_data_jpeg`. The
crops are scaled for display, so they do not hold the original 16-bit
intensities.

- **Metadata stays:** `images.images` records each source file name, well, site,
  split, size, and bit depth, and `source_url` points to BBBC039.
- **Measuring again:** download the BBBC039 `images.zip` and `masks.zip`, then
  follow "Rebuild Data Assets". The stored cp_measure numbers came from those
  originals.
- **The cover image:** it shows one field of view. The build reads it from
  `src/picture_thousand_records/assets/representative_field.jpg`.

## How cp_measure Is Used

[cp_measure](https://github.com/afermg/cp_measure) is a Python library that
computes CellProfiler's measurements without running CellProfiler. The project
pins version 0.2.0 in `uv.lock`.

- **Input:** each BBBC039 manual nucleus mask (decoded to one label per
  nucleus) and the DNA channel of the same image. Each image is scaled to 0-1
  by its own maximum first, so intensity values are relative to the brightest
  pixel of that image.
- **Call:** `cp_measure.bulk.get_core_measurements(legacy=True)`.
  `legacy=True` selects CellProfiler's original percentile convention for the
  intensity quartile features.
- **Output:** 271 measurements per nucleus (size and shape, Feret, Zernike,
  intensity, texture, granularity, and radial distribution), stored in
  `bbbc039.objects`. Three `AreaShape_NormalizedMoment` columns are always
  NaN, and 1.9% of nuclei have NaN texture values.
- **Feature space:** `features.cp_measure` keeps 263 of those columns (the
  AreaShape, Granularity, Intensity, RadialDistribution, and Texture columns),
  fills NaN with the column median, standardizes them, and reduces them to 16
  dimensions with PCA. The fused space joins these 16 numbers with 32 MorphEm
  PCA dimensions.

cp_measure does not segment. The masks come from the BBBC039 manual
annotations.

## Build

```sh
uv run picture-thousand-records build
```

This writes:

- `database.jpg`
- `database.duckdb` (git ignores it; the same file is inside `database.jpg`)
- `manifest.json`
- `DATABASE_README.md`
- `index.html`

## The 3D Maps

Each feature table has three 3-D maps of its vectors, and the page draws all
three side by side:

| Map | Columns | Method |
|---|---|---|
| PCA | `pca_x`, `pca_y`, `pca_z` | scikit-learn `PCA`, 3 components |
| UMAP | `umap_x`, `umap_y`, `umap_z` | `umap-learn`, Euclidean, `random_state=42` |
| t-SNE | `tsne_x`, `tsne_y`, `tsne_z` | `openTSNE`, Barnes-Hut, perplexity 30, `random_state=42` |

Searching for a nucleus retrieves its neighbors in the feature space (with the
HNSW indexes, see below) and marks the same seed and neighbors on all three
maps. That shows how each method treats the same neighborhood.

`features.embedding_quality` records how well each map keeps true neighbors
together, measured over all nuclei against exact nearest neighbors in the
feature space:

- `neighbor_recall`: the share of a nucleus's 6 true neighbors that are also
  among its 15 nearest points on the map.
- `distance_ratio`: the median map distance to a true neighbor divided by the
  median distance between random pairs (lower is better).
- `explained_variance`: for PCA, the share of variance in the 3 axes.

The three maps are independent. Each keeps its own view:

- **Rotate:** the maps turn slowly by default. A map pauses while your pointer
  is over it, while you drag, when it is off screen, and if your system asks
  for reduced motion. Untick "Slowly rotate all three" to stop them. Drag to
  turn one map yourself.
- **Zoom:** scroll the mouse wheel over a map to zoom toward the cursor. A
  trackpad pinch works too, and so does each map's Zoom slider. When a map is
  fully zoomed out and you scroll down, the scroll goes back to the page, so a
  map does not trap it. "Zoom to selection" centers a map on the searched
  nucleus and its 12 neighbors and zooms to fit them. Nothing zooms by itself.
- **Pan:** hold Shift and drag. A map turns around the point at the center of
  its view, so a zoomed region stays in place while it rotates.
- **Reset:** each map has a Reset button, and "Reset all views" resets all
  three.
- **Search:** click a point on any map. Hover to see the nucleus crop.

Each map scales its cloud so 95% of the points fit in its view. It pulls
the few far outliers toward the center so every nucleus stays on screen.

The page overlays the nuclei Cellpose missed on the PCA, UMAP, and t-SNE maps.
Those maps were computed from manual masks and DNA crops; the Cellpose verdict
was not part of the feature vector. If missed nuclei cluster, that suggests
they share measured shape, intensity, or learned visual neighborhoods. If they
scatter, the miss may depend on image context, touching cells, or segmentation
behavior that these nucleus-level features do not capture. cp_measure is
easiest to interpret through explicit size, shape, brightness, texture, and
radial intensity columns; MorphEm can group subtler crop appearance but is
harder to explain; Fused combines both, but you still need to query the source
columns and crops to decide which side explains a miss.

Rebuilding the coordinates takes several minutes, mostly the 3-D t-SNE. Run
`extract_feature_spaces.py` (see "Rebuild Data Assets") once the MorphEm matrix
is cached.

## Nearest-Neighbor Search

The database holds an HNSW index for the cp_measure and fused feature spaces.
DuckDB's [`vss`](https://duckdb.org/docs/current/core_extensions/vss) extension
builds the indexes when you run the build, and the page's search uses them. The
build needs network access once to install `vss`.

- **Why an index:** an index makes search faster. An exact scan compares a
  nucleus with every other one, which gets slow on large collections. HNSW
  links nuclei in layers, and a search moves from the sparse top layer down to
  the dense bottom layer. The answer is approximate, and the index takes
  space. At 19,565 nuclei an exact scan is already fast, so the index here is a
  demonstration.
- **Load:** run `INSTALL vss; LOAD vss;` and
  `SET hnsw_enable_experimental_persistence = true;` before you `ATTACH` the
  database. Without `vss`, the data still opens and the same queries run as an
  exact scan.
- **Query:** DuckDB uses an index only when the query vector is a constant.
  Copy the seed vector into a variable with `SET VARIABLE`, then use
  `getvariable()` in the `ORDER BY array_distance(...) LIMIT n` query. Run
  `EXPLAIN` to see `HNSW_INDEX_SCAN`.
- **Accuracy:** the build compares each index with an exact scan on about 200
  nuclei and stores the result in `manifest.json` (98% or better for the 12
  nearest neighbors in both indexed spaces; the exact figures vary a little
  from build to build).
- **MorphEm has no stored index.** Its 384-dimension index would add about
  33 MB, as much as the vectors themselves. MorphEm searches run the same SQL as
  an exact scan over all 19,565 vectors, which is still instant. The page's
  status line says which method ran.
- **Size:** the two stored indexes add about 11 MB to the database.
- **`features.knn_graph`:** the build also finds the 6 nearest neighbors of
  every nucleus in all three spaces, using temporary HNSW indexes (the
  MorphEm one is never stored). The page averages the 6 distances for each
  nucleus into its "neighbor density" (a short average distance means a dense
  neighborhood), ranks all nuclei, and colors the map from blue (sparse) to
  yellow (dense). None of the three maps preserves density, so this color shows
  something the position of a point does not.
- **Measurement precision:** `bbbc039.objects` stores the cp_measure values as
  32-bit `FLOAT` (about 7 significant digits) instead of `DOUBLE`. This saves
  about 35 MB.

## Run Locally

```sh
uv run poe run
```

Then open `http://localhost:8000/index.html`.

## Rebuild Data Assets

Download and extract the BBBC039 `images.zip`, `masks.zip`, and `metadata.zip`
files into `tools/bbbc039_data/raw/`. Then run:

```sh
uv run --group real_data python tools/bbbc039_data/extract_features.py
uv run --group morphem python tools/bbbc039_data/extract_feature_spaces.py
uv run --group real_data --group cellpose \
    python tools/bbbc039_data/match_cellpose.py
```

The first command runs cp_measure on every mask and writes the image-metadata,
object (with the 271 measurements), and crop parquet files. The package
assets keep only the JPEG crops. The full 16-bit crops go to
`tools/bbbc039_data/output/` because the MorphEm step needs them. The second command
reduces the cp_measure columns with PCA, computes MorphEm embeddings, and
writes the cp_measure, MorphEm, and fused parquet files. The third runs
[Cellpose](https://github.com/MouseLand/cellpose) (model `cpsam_v2`) on every
field and records, per manually annotated nucleus, whether a Cellpose mask
captured it (IoU ≥ 0.5); the build joins those columns into
`bbbc039.objects`. IoU means
[intersection over union](https://en.wikipedia.org/wiki/Jaccard_index): the
shared area divided by the total area covered by either mask. An IoU of 1 means
perfect overlap, and 0 means no overlap. The first Cellpose run downloads about
1 GB of model weights. On Apple Silicon the segmentation runs on the MPS GPU (a few
minutes); on CPU-only machines expect hours. Finished masks are cached under
`tools/bbbc039_data/output/cellpose_masks/`, so interrupted runs resume where
they stopped.

## Validate

```sh
uv run picture-thousand-records validate database.jpg
uv run pytest
```

## References

- [BBBC039](https://bbbc.broadinstitute.org/BBBC039)
- [F3](https://doi.org/10.1145/3749163)
- [JUMP-lite](https://arxiv.org/abs/2608.07632)
- [cp_measure](https://github.com/afermg/cp_measure) and its
  [paper](https://arxiv.org/abs/2507.01163) (Munoz et al., 2025)
- [CellProfiler](https://doi.org/10.1186/gb-2006-7-10-r100)
- [CaicedoLab/MorphEm](https://huggingface.co/CaicedoLab/MorphEm)
- [Cellpose](https://doi.org/10.1038/s41592-020-01018-x)
- [Cellpose-SAM](https://doi.org/10.1101/2025.04.28.651001)
- [Jaccard index / IoU](https://en.wikipedia.org/wiki/Jaccard_index)
- [DuckDB-Wasm](https://github.com/duckdb/duckdb-wasm)
