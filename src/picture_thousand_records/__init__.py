"""Build and open JPEG Warehouse artifacts."""

from picture_thousand_records.builder import build_artifacts
from picture_thousand_records.duckdb_access import connect_to_warehouse

__all__ = ["build_artifacts", "connect_to_warehouse"]
