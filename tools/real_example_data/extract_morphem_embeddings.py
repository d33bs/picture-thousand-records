"""Generate real MorphEm embeddings for each real segmented cell.

This is a one-off data-preparation script, not part of the installed
package. It loads the real per-object crops already produced by
`extract_features.py` (`output/real_object_crops.parquet`) and runs each
channel crop through the real, pretrained
[CaicedoLab/MorphEm](https://huggingface.co/CaicedoLab/MorphEm) model -- a
Vision Transformer Small trained with DINO on the CHAMMI-75 microscopy
dataset (MIT license) -- exactly as documented on its model card: each
channel is processed independently through
`SaturationNoiseInjector -> PerImageNormalize -> Resize(224, 224)`, then
`model.forward_features(...)["x_norm_clstoken"]` (384-d) is taken per
channel and the three channels' embeddings are concatenated into one
1152-d vector per cell (the model card's own "Bag of Channels" recipe).

It then projects those real 1152-d embeddings down to 2-D two independent,
standard ways, so the browser's cluster map can show both side by side:

1. [UMAP](https://umap-learn.readthedocs.io/) (McInnes, Healy & Melville,
   2018), via the reference `umap-learn` implementation, with a fixed
   `random_state` for reproducibility and the same Euclidean metric that
   `array_distance` uses for nearest-neighbor search in DuckDB -- producing
   the `umap_x`/`umap_y` columns.
2. A real k-nearest-neighbor graph (k=6) built by querying a real
   [HNSW](https://github.com/nmslib/hnswlib) index (Malkov & Yashunin,
   2016), via the reference `hnswlib` implementation, over the same
   embeddings, then laid out with a Fruchterman-Reingold force-directed
   layout (`networkx.spring_layout`, also seeded for reproducibility) --
   producing the `graph_x`/`graph_y` columns plus the `neighbor_object_number`/
   `distance` edge list written to `real_morphem_knn_graph.parquet`.

Neither projection is a redraw of the other: UMAP models the embedding as a
continuous manifold, while the HNSW graph shows the actual sparse
approximate-nearest-neighbor structure a real vector index would use for
retrieval -- comparing the two is itself a real sanity check that the two
independent, standard methods agree on the same cluster structure.

Run with:

    uv run --group morphem python3 tools/real_example_data/extract_morphem_embeddings.py
"""

from __future__ import annotations

from pathlib import Path

import hnswlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import umap
from torch import nn
from torchvision.transforms import v2
from transformers import AutoModel

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
CROPS_PATH = OUTPUT_DIR / "real_object_crops.parquet"
MODEL_NAME = "CaicedoLab/MorphEm"
CHANNEL_ORDER = ["DNA", "PH3", "cellbody"]
KNN_GRAPH_K = 6


# The following two transforms are copied verbatim from the model card's own
# "How to Get Started" example -- this is the exact preprocessing MorphEm
# was evaluated with, not a reimplementation.
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low: float = 200, high: float = 255) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


class PerImageNormalize(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


def _morphem_transform() -> v2.Compose:
    return v2.Compose(
        [
            SaturationNoiseInjector(),
            PerImageNormalize(),
            v2.Resize(size=(224, 224), antialias=True),
        ]
    )


def _embed_channel_crop(
    model: torch.nn.Module,
    transform: v2.Compose,
    crop: np.ndarray,
) -> np.ndarray:
    """Run one single-channel crop through MorphEm and return its 384-d
    CLS-token embedding. Processed one image at a time (N=1) so
    `PerImageNormalize` normalizes this image alone, matching its name.
    """

    tensor = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    tensor = transform(tensor).unsqueeze(1)  # (1, 1, 224, 224)
    with torch.no_grad():
        output = model.forward_features(tensor)
    return output["x_norm_clstoken"].squeeze(0).cpu().numpy()


def _build_knn_graph(
    object_numbers: np.ndarray,
    embedding_matrix: np.ndarray,
) -> tuple[pd.DataFrame, nx.Graph]:
    """Query a real HNSW index for each real cell's own k nearest real
    neighbors, using the reference `hnswlib` implementation (Malkov &
    Yashunin, 2016). Returns the edge list plus the resulting graph (for
    laying out with a force-directed algorithm).
    """

    dim = embedding_matrix.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(object_numbers), ef_construction=200, M=16)
    index.add_items(embedding_matrix, object_numbers)
    index.set_ef(50)

    k = min(KNN_GRAPH_K, len(object_numbers) - 1)
    labels, squared_distances = index.knn_query(embedding_matrix, k=k + 1)

    graph = nx.Graph()
    graph.add_nodes_from(int(n) for n in object_numbers)
    edge_rows = []
    for source, neighbor_labels, neighbor_sq_dists in zip(
        object_numbers, labels, squared_distances
    ):
        rank = 0
        for neighbor, squared_dist in zip(neighbor_labels, neighbor_sq_dists):
            if int(neighbor) == int(source):
                continue
            rank += 1
            if rank > k:
                break
            # hnswlib's "l2" space returns squared Euclidean distance; take
            # the square root so this is directly comparable to DuckDB's
            # array_distance, which returns (unsquared) Euclidean distance.
            distance = float(np.sqrt(squared_dist))
            edge_rows.append(
                {
                    "ObjectNumber": int(source),
                    "neighbor_object_number": int(neighbor),
                    "rank": rank,
                    "distance": distance,
                }
            )
            graph.add_edge(int(source), int(neighbor), weight=distance)

    return pd.DataFrame(edge_rows), graph


def main() -> None:
    if not CROPS_PATH.exists():
        raise SystemExit(f"{CROPS_PATH} not found -- run extract_features.py first.")

    crops = pd.read_parquet(CROPS_PATH)
    cells = crops[crops["object_type"] == "cells"]

    print(f"loading {MODEL_NAME} ...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    transform = _morphem_transform()

    object_numbers = sorted(cells["ObjectNumber"].unique())
    rows = []
    for i, object_number in enumerate(object_numbers):
        object_rows = cells[cells["ObjectNumber"] == object_number].set_index("channel")
        channel_embeddings = []
        for channel in CHANNEL_ORDER:
            crop_row = object_rows.loc[channel]
            crop = np.frombuffer(crop_row["pixel_data_raw"], dtype=np.uint8).reshape(
                crop_row["height"], crop_row["width"]
            )
            channel_embeddings.append(_embed_channel_crop(model, transform, crop))
        embedding = np.concatenate(channel_embeddings).astype(np.float32)
        rows.append(
            {
                "image_id": object_rows.iloc[0]["image_id"],
                "object_type": "cells",
                "ObjectNumber": int(object_number),
                "model": MODEL_NAME,
                "channels": CHANNEL_ORDER,
                "embedding_dim": len(embedding),
                "embedding": embedding,
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(object_numbers):
            print(f"  embedded {i + 1}/{len(object_numbers)} cells")

    features = pd.DataFrame(rows)

    print("projecting embeddings to 2-D with UMAP ...")
    embedding_matrix = np.stack(features["embedding"].to_numpy())
    # random_state fixes the layout for reproducibility; metric="euclidean"
    # matches array_distance, the same distance the browser's nearest-
    # neighbor search runs on, so proximity in the 2-D plot reflects the
    # same real relationships "Find similar" surfaces.
    reducer = umap.UMAP(n_components=2, metric="euclidean", random_state=42)
    projection = reducer.fit_transform(embedding_matrix)
    features["umap_x"] = projection[:, 0].astype(np.float32)
    features["umap_y"] = projection[:, 1].astype(np.float32)

    print(f"building a real k={KNN_GRAPH_K} HNSW nearest-neighbor graph ...")
    object_numbers = features["ObjectNumber"].to_numpy()
    knn_graph, graph = _build_knn_graph(object_numbers, embedding_matrix)

    print("laying out the k-NN graph with a force-directed layout ...")
    layout = nx.spring_layout(graph, seed=42)
    features["graph_x"] = features["ObjectNumber"].map(
        lambda obj: float(layout[int(obj)][0])
    ).astype(np.float32)
    features["graph_y"] = features["ObjectNumber"].map(
        lambda obj: float(layout[int(obj)][1])
    ).astype(np.float32)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "real_morphem_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"wrote {len(features)} rows -> {out_path}")

    graph_out_path = OUTPUT_DIR / "real_morphem_knn_graph.parquet"
    knn_graph.to_parquet(graph_out_path, index=False)
    print(f"wrote {len(knn_graph)} rows -> {graph_out_path}")


if __name__ == "__main__":
    main()
