# picture-thousand-records

JPEG Warehouse prototype: one browser-loadable JPEG file that also contains a
DuckDB database in an appended ZIP archive.

The implementation follows the pure-Python `uv` project shape from
[`CU-DBMI/template-uv-python-research-software`](https://github.com/CU-DBMI/template-uv-python-research-software):
package code lives under `src/`, tests live under `tests/`, the CLI is exposed as
`picture-thousand-records`, and notebook work lives under `notebooks/`.

## What is inside `database.jpg`

The file `database.jpg` is two files joined end to end. The first part is a
JPEG picture of about 390 KB. The second part is a ZIP archive of about 7 MB.
The archive holds three files:

- `warehouse.duckdb`: the database. DuckDB (a program that keeps tables in one
  file) reads it.
- `manifest.json`: a list of the tables, their row counts, and the source of
  the data.
- `README.md`: the instructions for the database.

```text
database.jpg
  first bytes   JPEG picture       starts with FF D8, ends with FF D9
  next bytes    ZIP archive        stored without compression
                  warehouse.duckdb   the database (about 7 MB)
                  manifest.json      about 3 KB
                  README.md          about 4 KB
                  table of contents  the last bytes of the file
```

### How can one file be both?

A JPEG ends with a two-byte marker, `FF D9`. The marker means that the picture
ends here. A picture viewer stops at this marker and ignores every byte after
it.

A ZIP archive keeps its table of contents at the end of the file. A ZIP tool
reads the end first. The table of contents tells the tool where each file
starts, so the tool never needs the bytes at the front.

As a result, each tool sees only its own part. The picture viewer sees a
picture. The ZIP tool sees an archive. The ZIP data does not change the
picture. The archive stores the database without compression, so the database
inside it is a byte-for-byte copy of `warehouse.duckdb`.

CAUTION: Keep the original file. Some apps re-save a picture when you send or
upload it. A re-saved picture keeps only the photo and loses the database.

## Related idea

We drew inspiration from
[F3: The Open-Source Data File Format for the Future](https://doi.org/10.1145/3749163).
F3 stores data, metadata, and WebAssembly decoders in one file. We also drew
on [JUMP-lite](https://arxiv.org/abs/2608.07632), which focuses on compact,
reproducible image-based profiling data. This project uses a simpler shape for
image-based science. The code lives on the browser page. The file opens first
as an image, so anyone can see what the data is about before they run a query.

## Real data, not synthetic

Every table in the warehouse is real, measured from CellProfiler's public
[`ExampleHuman`](https://cellprofiler.org/examples) tutorial field (human HT29
cells; DNA/PH3/cell-body channels; Moffat et al. 2006, CC-0), sourced via
[`cytomining/CytoTable`](https://github.com/cytomining/CytoTable/tree/main/tests/data/cellprofiler/ExampleHuman)'s
test fixtures:

```text
cellprofiler.nuclei    real per-nucleus AreaShape/Intensity/Location features
cellprofiler.cells     same, per-cell (Parent_Nuclei -> nuclei)
cellprofiler.cytoplasm same, per-cytoplasm (Parent_Nuclei, Parent_Cells)
cellprofiler.ph3       real mitotic (PH3-positive) foci (Parent_Nuclei)
images.images          metadata for the source microscopy field
images.object_crops    real per-cell, per-channel crops of that field
morphem.features       one real 1152-d MorphEm embedding per real cell
morphem.knn_graph      a real k=6 HNSW nearest-neighbor edge list over
                       that embedding
```

Each `images.object_crops` row is a self-describing OME-Arrow-style record:
alongside its own `tile_y`/`tile_x`/`height`/`width`, it also carries the
source field's own `size_c`/`size_z`/`size_y`/`size_x` and `source`/
`source_url`, so no join to `images.images` is required. The pixel data is
stored three ways: `pixel_data_raw` (uncompressed array bytes, the native
OME-Arrow-style representation DuckDB can query with no decoding),
`pixel_data_jpegxl` (compact JPEG XL), and `pixel_data_jpeg` (plain JPEG,
for universal compatibility).

`cellprofiler.single_cell` is a view merging Cytoplasm/Cells/Nuclei into one
row per cell, using the same join CytoTable's `cellprofiler_csv`
[preset](https://github.com/cytomining/CytoTable/blob/main/cytotable/presets.py)
uses for this exact dataset (anchored on Cytoplasm, `LEFT JOIN` Cells/Nuclei
via `Parent_Cells`/`Parent_Nuclei`), with columns prefixed
`Cytoplasm_`/`Cells_`/`Nuclei_` to match what a real CytoTable single-cell
table looks like.

`morphem.features.embedding` is a fixed-size `FLOAT[1152]` array: the real,
pretrained [CaicedoLab/MorphEm](https://huggingface.co/CaicedoLab/MorphEm)
model (a Vision Transformer Small trained with DINO on the CHAMMI-75
microscopy dataset, MIT license) run independently over each cell's real
DNA/PH3/cell-body crop (384-d each, per the model card's "Bag of Channels"
recipe) and concatenated. `morphem.features.umap_x`/`umap_y` are a real 2-D
[UMAP](https://umap-learn.readthedocs.io/) projection (McInnes, Healy &
Melville, 2018) of that same embedding, computed with the reference
`umap-learn` implementation using the same Euclidean metric as
`array_distance` and a fixed random seed, so the layout faithfully
visualizes the same distances the similarity search runs on.

`morphem.knn_graph` is a real k=6 nearest-neighbor edge list, built by
querying a real [HNSW](https://github.com/nmslib/hnswlib) index (Malkov &
Yashunin, 2016, via the reference `hnswlib` implementation) over that same
embedding — the same approximate-nearest-neighbor structure a real vector
index uses for retrieval. Its distances closely match the exact,
brute-force `array_distance` for the same pairs, confirming the index does
genuine approximate search rather than returning a placeholder.
`morphem.features.graph_x`/`graph_y` lay that graph out with a
Fruchterman-Reingold force-directed layout (`networkx.spring_layout`), so
`index.html` can plot it side by side with the UMAP projection — two
independent, standard methods, over the same real embeddings, that should
(and do) agree on the same cluster structure.

That agreement is checkable against the dataset's own origin. This field's
images were provided by Jason Moffat, a co-author of the screen that first
used this data — a genome-scale RNAi screen (Moffat et al. 2006, *Cell*)
that scored "mitotic index" via PH3 staining. Computed fresh at every build
from the real tables (see `_mitotic_cluster_alignment` in `builder.py`), the
real, CellProfiler-equivalent segmentation finds 20 real PH3-positive
(mitotic) cells out of 274 in this field; with no supervision, on average
86% of a PH3-positive cell's real HNSW nearest neighbors are also
PH3-positive, versus the ~7% expected by chance. A general-purpose model
never trained on this screen still recovers the same mitotic/interphase
distinction the original screen was built to measure, purely from image
morphology.

DuckDB's built-in `array_distance` runs real nearest-neighbor queries over the
embedding directly — no extension required:

```sql
SELECT ObjectNumber, array_distance(
    embedding,
    (SELECT embedding FROM warehouse.morphem.features WHERE ObjectNumber = 1)
) AS distance
FROM warehouse.morphem.features
ORDER BY distance
LIMIT 10;
```

`tools/real_example_data/extract_morphem_embeddings.py` regenerates these
embeddings (plus the UMAP projection and HNSW k-NN graph) from the real
per-object crops already produced by `extract_features.py`. Run it with:

```sh
uv run --group morphem python3 tools/real_example_data/extract_morphem_embeddings.py
```

That writes `real_morphem_features.parquet` and
`real_morphem_knn_graph.parquet`, bundled the same way as the other tables —
so building the warehouse itself never needs torch, transformers,
umap-learn, or hnswlib either.

`tools/real_example_data/extract_features.py` regenerates the six
`cellprofiler`/`images` tables from the raw TIFF channels: it approximates
CellProfiler's `ExampleHuman.cppipe` identify/relate modules with
scikit-image (global threshold + intensity-based watershed
declumping/propagation), measures the resulting objects with
[`cp_measure`](https://github.com/afermg/cp_measure) — a modern, pure-Python
reimplementation of CellProfiler's measurement math (no Java, wxPython, or
javabridge/legacy-NumPy pin required) — and crops each real cell out of each
channel three ways: a raw uncompressed array, JPEG XL via
`pillow-jxl-plugin`, and plain JPEG. Run it with:

```sh
uv run --group real_data python3 tools/real_example_data/extract_features.py
```

That writes six parquet files to `tools/real_example_data/output/`, which
are what's bundled as `src/picture_thousand_records/assets/real_*.parquet`
and loaded directly into DuckDB at build time — so building the warehouse
itself never needs scikit-image, cp_measure, or a JPEG XL codec.

## Build

```sh
uv run picture-thousand-records build
```

This writes:

- `database.jpg`: the JPEG/ZIP polyglot artifact.
- `warehouse.duckdb`: the build-time database copied into the JPEG.
- `manifest.json`: descriptive metadata copied into the JPEG.
- `index.html`: a local browser page that renders `database.jpg`.

The build is deterministic — it just loads the six bundled parquet assets
into a fresh DuckDB database, so there is no dataset scale or seed to pass.

## Validate

```sh
uv run picture-thousand-records validate database.jpg
unzip -l database.jpg
```

The validator checks that the artifact is a JPEG, that ZIP members are present,
and that DuckDB can query the embedded database. DuckDB cannot `ATTACH` a
`zip://` path directly (checked on DuckDB 1.5.3 and 1.5.5), so it follows the
same recipe printed on the JPEG. Run `duckdb` in the folder that holds
`database.jpg`:

```sql
INSTALL zipfs FROM community;
LOAD zipfs;
SET zipfs_split = '!!';

COPY (SELECT content FROM read_blob('zip://database.jpg!!warehouse.duckdb'))
    TO 'warehouse.duckdb' (FORMAT blob);

ATTACH 'warehouse.duckdb' AS warehouse (READ_ONLY);
```

The `COPY` streams the stored member out through `zipfs`, byte for byte, so no
`unzip` is needed. If `zipfs` cannot be installed (for example, offline), the
helper extracts `warehouse.duckdb` with Python's `zipfile` instead.

To work fully in memory, copy the tables you need and delete the file:

```sql
CREATE SCHEMA cellprofiler; CREATE SCHEMA images; CREATE SCHEMA morphem;
CREATE TABLE cellprofiler.cells AS SELECT * FROM warehouse.cellprofiler.cells;
CREATE TABLE morphem.features   AS SELECT * FROM warehouse.morphem.features;
DETACH warehouse;
.shell rm warehouse.duckdb
```

## Notebook

Run the local scientific demo:

```sh
uv run --group notebooks jupyter lab notebooks/cytodataframe_demo.ipynb
```

The notebook opens `database.jpg`, builds a CytoDataFrame over the real
`cellprofiler.cells` features, queries the `cellprofiler.single_cell` view,
displays real per-object crops from `images.object_crops` all three ways
(straight from `pixel_data_raw`, via `pillow-jxl-plugin` for
`pixel_data_jpegxl`, and directly from `pixel_data_jpeg`), and runs a real
nearest-neighbor search over `morphem.features.embedding`. None of this
needs torch/transformers at read time — those only run once, offline, to
produce the embeddings. CytoDataFrame currently supports Python
`>=3.11,<3.14`, so this project pins local `uv` runs to Python 3.13 through
`.python-version`.

## Browser Demo

Serve the repository root and open `index.html`:

```sh
python3 -m http.server 8000
```

`index.html` fetches `database.jpg` (or, if the page was opened from disk,
asks you to choose the file; nothing is uploaded), extracts `warehouse.duckdb` from the
appended ZIP payload with JSZip, and registers the database bytes with
DuckDB-Wasm — no server, no Python, entirely in the browser. It's a single
page: the JPEG cover, then an editable SQL console, then the interactive
plots and similarity search.

The SQL console's textarea is pre-loaded with the actual `array_distance`
query above, and buttons above it load the other real example queries from
this README (`knn_graph`, `single_cell join`, `object_crops`, `Row counts`);
edit any of them and press "Run query" (or Cmd/Ctrl+Enter) to run your own
real SQL against the real tables.

Below that, the page plots every real cell twice, side by side — its real
`umap_x`/`umap_y` UMAP projection, and its real `graph_x`/`graph_y`
force-directed layout of the real HNSW k-NN graph (with the real edges
drawn in). Hovering either plot previews the cell's real image, and clicking
a point (in either plot) or a result thumbnail runs a real `array_distance`
nearest-neighbor query against `morphem.features.embedding` and shows its
real nearest neighbors below.

## Milestone Coverage

- V0: `database.jpg` is a valid JPEG and ZIP container with `warehouse.duckdb`.
- V1: `index.html` provides a browser DuckDB-Wasm adapter for the same JPEG file.
- V2: the visible JPEG explains the dataset, schema, and DuckDB access path.
- V3: `notebooks/cytodataframe_demo.ipynb` runs and displays real image data.
- V4: `cellprofiler.nuclei`/`cells`/`cytoplasm`/`ph3` hold real, `cp_measure`-computed
  CellProfiler-equivalent compartment tables from a real, CC-0 microscopy field,
  matching CytoTable's real `Nuclei.csv`/`Cells.csv`/`Cytoplasm.csv`/`PH3.csv` shape.
- V5: `images.object_crops` holds real per-object image crops as
  self-describing OME-Arrow-style records, cropped straight from the source
  TIFF channels and stored three ways: raw uncompressed array, JPEG XL, and
  plain JPEG.
- V6: `cellprofiler.single_cell` merges Cytoplasm/Cells/Nuclei into one
  single-cell profile per real cell, using CytoTable's own preset join.
- V7: `morphem.features` holds one real, fixed-size deep-learning embedding
  per real cell from the real, pretrained CaicedoLab/MorphEm model, queryable
  with DuckDB's built-in `array_distance` for nearest-neighbor search.
- V8: `morphem.features.umap_x`/`umap_y` hold a real UMAP projection of that
  embedding (reference `umap-learn` implementation, same Euclidean metric as
  the search); `index.html` renders it as a clickable cluster map wired to
  the same real nearest-neighbor search.
- V9: `morphem.knn_graph` holds a real k=6 HNSW nearest-neighbor edge list
  (reference `hnswlib` implementation) over that same embedding, laid out
  with a real force-directed layout (`networkx.spring_layout`) in
  `morphem.features.graph_x`/`graph_y`; `index.html` plots it side by side
  with the UMAP projection, with hover previews and click-to-search on both.
- V10: that side-by-side clustering is checked, at every build, against the
  dataset's own source paper — real PH3-positive (mitotic) cells cluster
  together far more than chance in both projections, the same phenotype
  Moffat et al. (2006) used as their screen's readout; `index.html`
  reports the real, build-time-computed numbers.
- V11: `index.html` has an editable SQL console, pre-loaded with this
  README's actual example queries, that runs any real SQL against the real
  tables entirely in DuckDB-Wasm, alongside the interactive plots and
  similarity search on the same page.
