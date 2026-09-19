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
    assert "connect_to_warehouse" in source
    assert "cellprofiler.cells" in source
    assert "images.object_crops" in source
    assert "pixel_data_raw" in source
    assert "pixel_data_jpegxl" in source
    assert "pixel_data_jpeg" in source
    assert "CytoDataFrame" in source
    assert "Image_FileName_DNA" in source
    assert "data_context_dir" in source
    assert "render_whole_image" in source
    assert "tempfile.mkdtemp" in source
    assert "pillow_jxl" in source
    assert "cellprofiler.single_cell" in source
    assert "morphem.features" in source
    assert "array_distance" in source
    assert "media.jpeg_images" not in source
    assert "ome_arrow" not in source
    assert "images.tiles" not in source
