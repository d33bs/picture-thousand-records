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

    The preferred path is DuckDB's `zipfs` extension. If the local DuckDB build
    cannot attach a database through `zipfs`, this helper extracts the embedded
    member to a temporary directory and attaches that database read-only.
    """

    path = Path(jpeg_path).resolve()
    con = duckdb.connect()
    try:
        _attach_zipfs(con, path, alias)
        return WarehouseConnection(connection=con, mode="zipfs")
    except duckdb.Error:
        extracted = _extract_database(path, temp_dir)
        con.execute(f"ATTACH '{extracted.as_posix()}' AS {alias} (READ_ONLY)")
        return WarehouseConnection(
            connection=con,
            mode="extracted",
            extracted_database=extracted,
        )


def _attach_zipfs(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    alias: str,
) -> None:
    con.execute("INSTALL zipfs FROM community")
    con.execute("LOAD zipfs")
    con.execute("SET zipfs_split = '!!'")
    con.execute(
        f"ATTACH 'zip://{path.as_posix()}!!warehouse.duckdb' AS {alias} (READ_ONLY)"
    )


def _extract_database(path: Path, temp_dir: str | Path | None) -> Path:
    destination_dir = (
        Path(temp_dir)
        if temp_dir is not None
        else Path(tempfile.mkdtemp(prefix="jpeg-warehouse-"))
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "warehouse.duckdb"
    with (
        zipfile.ZipFile(path) as archive,
        archive.open("warehouse.duckdb") as source,
    ):
        destination.write_bytes(source.read())
    return destination
