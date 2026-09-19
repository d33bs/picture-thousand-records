"""Command-line interface for picture-thousand-records."""

from __future__ import annotations

import json
from pathlib import Path

import fire

from picture_thousand_records.builder import build_artifacts
from picture_thousand_records.validation import validate_artifact


class PictureThousandRecordsCLI:
    """Build and validate the JPEG Warehouse prototype."""

    def build(self, output_dir: str = ".") -> None:
        """Build `database.jpg`, `warehouse.duckdb`, and `index.html`."""

        paths = build_artifacts(output_dir=output_dir)
        print(paths.jpeg)

    def validate(self, path: str = "database.jpg") -> None:
        """Validate a JPEG Warehouse artifact."""

        report = validate_artifact(Path(path))
        print(json.dumps(report, indent=2))


def trigger() -> None:
    """Run the CLI."""

    fire.Fire(PictureThousandRecordsCLI)
