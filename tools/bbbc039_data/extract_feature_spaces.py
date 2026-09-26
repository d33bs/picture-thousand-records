"""Build BBBC039 cp_measure, MorphEm, and fused feature spaces.

Each feature space also gets three 3-D maps for the page: PCA (linear), UMAP,
and t-SNE (openTSNE, Barnes-Hut). A small table records how well each map keeps
the true nearest neighbors of every nucleus close together.

The nearest-neighbor indexes are not built here. `builder.py` builds them
inside DuckDB (the `vss` extension) when it assembles the database.

Run with:

    uv run --group morphem python tools/bbbc039_data/extract_feature_spaces.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
from openTSNE import TSNE
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoModel

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
PACKAGE_ASSETS_DIR = HERE.parent.parent / "src" / "picture_thousand_records" / "assets"
OBJECTS_PATH = OUTPUT_DIR / "bbbc039_objects.parquet"
CROPS_PATH = OUTPUT_DIR / "object_crops.parquet"
MORPHEM_MATRIX_PATH = OUTPUT_DIR / "morphem_matrix.npy"
MODEL_NAME = "CaicedoLab/MorphEm"
CP_MEASURE_DIM = 16
MORPHEM_DIM = 384
FUSED_MORPHEM_DIM = 32
FUSED_DIM = CP_MEASURE_DIM + FUSED_MORPHEM_DIM
BATCH_SIZE = 64
MAP_METHODS = ("pca", "umap", "tsne")
QUALITY_NEIGHBORS = 6  # true neighbors of each nucleus, found by exact search
QUALITY_MAP_NEIGHBORS = 15  # how far out on the map a true neighbor may sit
TSNE_PERPLEXITY = 30


class PerImageNormalize(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm(x)


def _resize_crop(crop: np.ndarray) -> np.ndarray:
    image = Image.fromarray(crop.astype(np.uint8), mode="L")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.float32)


def _load_crop(row: pd.Series) -> np.ndarray:
    raw = np.frombuffer(row["pixel_data_raw"], dtype=np.uint16)
    crop = raw.reshape(int(row["height"]), int(row["width"]))
    low, high = np.percentile(crop, [0.5, 99.5])
    if high <= low:
        high = float(crop.max() or 1)
        low = float(crop.min())
    scaled = np.clip((crop.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _cp_measure_matrix(objects: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    prefixes = (
        "AreaShape_",
        "Granularity_",
        "Intensity_",
        "RadialDistribution_",
        "Texture_",
    )
    columns = [column for column in objects.columns if column.startswith(prefixes)]
    matrix = objects[columns].replace([np.inf, -np.inf], np.nan)
    medians = matrix.median(numeric_only=True).fillna(0)
    matrix = matrix.fillna(medians)
    scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype=np.float32))
    dim = min(CP_MEASURE_DIM, scaled.shape[1], len(objects) - 1)
    pca = PCA(n_components=dim, random_state=42)
    return pca.fit_transform(scaled).astype(np.float32), columns


def _morphem_matrix(crops: pd.DataFrame, object_numbers: list[int]) -> np.ndarray:
    if MORPHEM_MATRIX_PATH.exists():
        print(f"loading cached MorphEm matrix from {MORPHEM_MATRIX_PATH} ...")
        return np.load(MORPHEM_MATRIX_PATH)

    print(f"loading {MODEL_NAME} ...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    normalizer = PerImageNormalize()
    by_object = crops.set_index("ObjectNumber")
    embeddings = []
    for start in range(0, len(object_numbers), BATCH_SIZE):
        batch_numbers = object_numbers[start : start + BATCH_SIZE]
        batch = []
        for object_number in batch_numbers:
            crop = _load_crop(by_object.loc[object_number])
            crop[crop == 255] = np.random.default_rng(object_number).integers(
                200, 256, size=(crop == 255).sum(), dtype=np.uint8
            )
            batch.append(_resize_crop(crop))
        tensor = torch.from_numpy(np.stack(batch)).unsqueeze(1)
        tensor = normalizer(tensor)
        with torch.no_grad():
            output = model.forward_features(tensor)
        embeddings.append(output["x_norm_clstoken"].cpu().numpy().astype(np.float32))
        done = min(start + BATCH_SIZE, len(object_numbers))
        if done % 1024 < BATCH_SIZE or done == len(object_numbers):
            print(f"  embedded {done}/{len(object_numbers)} nuclei")
    matrix = np.vstack(embeddings).astype(np.float32)
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.save(MORPHEM_MATRIX_PATH, matrix)
    return matrix


def _project(
    frame: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[float]]:
    """Three 3-D maps of the same vectors: PCA, UMAP, and t-SNE."""

    pca = PCA(n_components=3, random_state=42)
    maps = {"pca": pca.fit_transform(matrix)}
    print("  umap ...")
    maps["umap"] = umap.UMAP(
        n_components=3, metric="euclidean", random_state=42
    ).fit_transform(matrix)
    print("  t-SNE (openTSNE, Barnes-Hut) ...")
    maps["tsne"] = np.asarray(
        TSNE(
            n_components=3,
            perplexity=TSNE_PERPLEXITY,
            metric="euclidean",
            negative_gradient_method="bh",
            random_state=42,
            n_jobs=-1,
        ).fit(matrix)
    )
    for method, projection in maps.items():
        for axis, name in enumerate("xyz"):
            frame[f"{method}_{name}"] = projection[:, axis].astype(np.float32)
    return frame, maps, [float(v) for v in pca.explained_variance_ratio_]


def _embedding_quality(
    feature_space: str,
    matrix: np.ndarray,
    maps: dict[str, np.ndarray],
    pca_variance: list[float],
) -> list[dict[str, object]]:
    """How well does each map keep true neighbors together?

    True neighbors are the exact nearest neighbors in the feature space.
    `neighbor_recall` is the share of them that are also among a point's
    nearest points on the map. `distance_ratio` is the median map distance to a
    true neighbor divided by the median map distance between random pairs
    (lower is better).
    """

    _, exact = (
        NearestNeighbors(n_neighbors=QUALITY_NEIGHBORS + 1)
        .fit(matrix)
        .kneighbors(matrix)
    )
    exact = np.array(
        [[j for j in row if j != i][:QUALITY_NEIGHBORS] for i, row in enumerate(exact)]
    )
    rng = np.random.default_rng(0)
    random_pairs = rng.integers(0, len(matrix), (200_000, 2))
    rows = []
    for method, projection in maps.items():
        _, on_map = (
            NearestNeighbors(n_neighbors=QUALITY_MAP_NEIGHBORS + 1)
            .fit(projection)
            .kneighbors(projection)
        )
        on_map = on_map[:, 1:]
        kept = np.mean(
            [np.isin(exact[i], on_map[i]).mean() for i in range(len(matrix))]
        )
        sources = np.repeat(np.arange(len(matrix)), exact.shape[1])
        near = np.linalg.norm(projection[sources] - projection[exact.ravel()], axis=1)
        random = np.linalg.norm(
            projection[random_pairs[:, 0]] - projection[random_pairs[:, 1]], axis=1
        )
        rows.append(
            {
                "feature_space": feature_space,
                "method": method,
                "n_components": 3,
                "neighbor_recall": float(kept),
                "distance_ratio": float(np.median(near) / np.median(random)),
                "explained_variance": (
                    float(sum(pca_variance)) if method == "pca" else float("nan")
                ),
            }
        )
        print(
            f"  {method:5s} neighbor recall {kept:.1%}, "
            f"distance ratio {rows[-1]['distance_ratio']:.3f}"
        )
    return rows


def _feature_frame(
    objects: pd.DataFrame,
    matrix: np.ndarray,
    feature_space: str,
    vector_column: str,
    model: str,
) -> pd.DataFrame:
    base_columns = [
        "ObjectNumber",
        "ImageNumber",
        "LocalObjectNumber",
        "image_id",
        "split",
        "plate",
        "Location_Center_X",
        "Location_Center_Y",
        "AreaShape_Area",
        "Intensity_MeanIntensity_DNA",
    ]
    frame = objects[base_columns].copy()
    frame.insert(3, "feature_space", feature_space)
    frame["model"] = model
    frame["embedding_dim"] = matrix.shape[1]
    frame[vector_column] = [row.astype(np.float32) for row in matrix]
    return frame


def _write_table(frame: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{name}.parquet"
    package_path = PACKAGE_ASSETS_DIR / f"{name}.parquet"
    frame.to_parquet(out_path, index=False)
    frame.to_parquet(package_path, index=False)
    print(f"wrote {len(frame)} rows, {len(frame.columns)} columns -> {out_path}")


def main() -> None:
    objects = (
        pd.read_parquet(OBJECTS_PATH).sort_values("ObjectNumber").reset_index(drop=True)
    )
    crops = (
        pd.read_parquet(CROPS_PATH).sort_values("ObjectNumber").reset_index(drop=True)
    )
    object_numbers = objects["ObjectNumber"].astype(int).to_list()

    print("building cp_measure feature matrix ...")
    cp_matrix, cp_columns = _cp_measure_matrix(objects)
    print(f"  used {len(cp_columns)} measurement columns")

    print("building MorphEm feature matrix ...")
    morphem_matrix = _morphem_matrix(crops, object_numbers)

    print("building fused feature matrix ...")
    morphem_scaled = StandardScaler().fit_transform(morphem_matrix)
    morphem_reduced = PCA(
        n_components=FUSED_MORPHEM_DIM,
        random_state=42,
    ).fit_transform(morphem_scaled)
    fused_matrix = np.hstack(
        [
            StandardScaler().fit_transform(cp_matrix),
            StandardScaler().fit_transform(morphem_reduced),
        ]
    ).astype(np.float32)

    spaces = [
        (
            "cp_measure",
            cp_matrix,
            "measurement_vector",
            "cp_measure (PCA of the CellProfiler measurements)",
        ),
        ("morphem", morphem_matrix, "embedding", MODEL_NAME),
        ("fused", fused_matrix, "fused_embedding", "cp_measure + MorphEm"),
    ]
    quality: list[dict[str, object]] = []
    for feature_space, matrix, vector_column, model in spaces:
        print(f"projecting {feature_space} features ...")
        feature_frame = _feature_frame(
            objects,
            matrix,
            feature_space,
            vector_column,
            model,
        )
        feature_frame, maps, pca_variance = _project(feature_frame, matrix)
        _write_table(feature_frame, f"features_{feature_space}")
        quality.extend(_embedding_quality(feature_space, matrix, maps, pca_variance))
    _write_table(pd.DataFrame(quality), "embedding_quality")


if __name__ == "__main__":
    main()
