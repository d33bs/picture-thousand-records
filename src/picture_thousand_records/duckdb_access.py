"""DuckDB helpers for opening a JPEG Warehouse."""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class WarehouseConnection:
    """A DuckDB connection plus the access mode that succeeded."""

    connection: duckdb.DuckDBPyConnection
    mode: str
    extracted_database: Path | None = None


def connect_to_warehouse(
    jpeg_path: str | Path,
    *,
    alias: str = "warehouse",
    temp_dir: str | Path | None = None,
) -> WarehouseConnection:
    """Attach the embedded DuckDB database from a JPEG Warehouse artifact.

    DuckDB cannot `ATTACH` a `zip://` path directly (checked on 1.5.x), so the
    preferred path is the SQL-only recipe in the JPEG's README: stream the
    stored member out through the `zipfs` extension with `read_blob`, then
    attach that copy read-only. If `zipfs` cannot be installed (for example,
    offline), the member is extracted with Python's `zipfile` instead.
    """

    path = Path(jpeg_path).resolve()
    con = duckdb.connect()
    destination = _destination(temp_dir)
    try:
        _copy_member_with_zipfs(con, path, destination)
        mode = "zipfs"
    except duckdb.Error:
        _extract_database(path, destination)
        mode = "extracted"
    con.execute(f"ATTACH '{destination.as_posix()}' AS {alias} (READ_ONLY)")
    return WarehouseConnection(
        connection=con,
        mode=mode,
        extracted_database=destination,
    )


def _destination(temp_dir: str | Path | None) -> Path:
    destination_dir = (
        Path(temp_dir)
        if temp_dir is not None
        else Path(tempfile.mkdtemp(prefix="jpeg-warehouse-"))
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / "warehouse.duckdb"


def _copy_member_with_zipfs(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    destination: Path,
) -> None:
    con.execute("INSTALL zipfs FROM community")
    con.execute("LOAD zipfs")
    con.execute("SET zipfs_split = '!!'")
    con.execute(
        "COPY (SELECT content FROM read_blob(?)) "
        f"TO '{destination.as_posix()}' (FORMAT blob)",
        [f"zip://{path.as_posix()}!!warehouse.duckdb"],
    )


def _extract_database(path: Path, destination: Path) -> None:
    with (
        zipfile.ZipFile(path) as archive,
        archive.open("warehouse.duckdb") as source,
    ):
        destination.write_bytes(source.read())
