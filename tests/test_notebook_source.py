from __future__ import annotations

import json
from pathlib import Path


def test_notebook_queries_embedded_images() -> None:
    notebook = json.loads(Path("notebooks/cytodataframe_demo.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "connect_to_database" in source
    assert "database.bbbc039.objects" in source
    assert "images.object_crops" in source
    assert "pixel_data_jpeg" in source
    assert "CytoDataFrame" in source
    assert "Metadata_ImageID" in source
    assert "Image_DNA" in source
    assert "jpeg_bytes_to_ome_arrow" in source
    assert '"type": "ome.arrow"' in source
    assert "data_context_dir" not in source
    assert "Image_FileName_DNA" not in source
    assert "jpeg_to_data_url" not in source
    assert "to_html" not in source
    assert "base64" not in source
    assert "tempfile.mkdtemp" in source
    assert "pillow_jxl" not in source
    assert "pixel_data_raw" not in source
    assert "source_images" not in source
    assert "features.cp_measure" in source
    assert "features.morphem" in source
    assert "features.fused" in source
    assert "features.knn_graph" in source
    assert "array_distance" in source
    assert "ExampleHuman" not in source
    assert "cellprofiler.cells" not in source
    assert "media.jpeg_images" not in source
    assert "images.tiles" not in source
