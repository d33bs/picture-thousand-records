"""Build and open JPEG Database artifacts."""

from picture_thousand_records.builder import build_artifacts
from picture_thousand_records.duckdb_access import connect_to_database

__all__ = ["build_artifacts", "connect_to_database"]
