"""Acquisition, audit, and chronological splitting for the Beijing air-quality data.

The UCI archive (dataset 501) holds hourly readings from 12 stations between
2013-03-01 00:00 and 2017-02-28 23:00. Timestamps are local Beijing time
(UTC+8, no daylight saving) and are kept timezone-naive.
"""

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from preprocessing import RAW_DATA_DIR

AIR_QUALITY_URL = (
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
)
# SHA-256 of the archive downloaded on 2026-10-22 (8,192,212 bytes).
AIR_QUALITY_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
STATION_ARCHIVE = "PRSA2017_Data_20130301-20170228.zip"
TIMESTAMP_PARTS = ["year", "month", "day", "hour"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_air_quality(
    raw_dir: Path = RAW_DATA_DIR,
    *,
    url: str = AIR_QUALITY_URL,
    sha256: str = AIR_QUALITY_SHA256,
) -> Path:
    """Download the UCI archive once and verify its checksum.

    Args:
        raw_dir: Raw-data root; the archive is stored at
            ``raw_dir / "beijing_air_quality" / "uci_501.zip"``.
        url: Source URL. Any scheme supported by ``urllib`` works.
        sha256: Expected hex digest of the archive.

    Returns:
        Path to the verified archive. An existing file is reused without network
        access.

    Raises:
        ValueError: If the existing or downloaded file does not match ``sha256``.
            A mismatching download is deleted; an existing file is left in place
            for inspection.
    """
    path = Path(raw_dir) / "beijing_air_quality" / "uci_501.zip"
    if path.exists():
        if _sha256(path) != sha256:
            raise ValueError(f"SHA-256 of existing {path} does not match {sha256}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    urlretrieve(url, partial)
    digest = _sha256(partial)
    if digest != sha256:
        partial.unlink()
        raise ValueError(f"SHA-256 of {url} is {digest}, expected {sha256}")
    partial.replace(path)
    return path


def read_stations(archive_path: Path) -> pd.DataFrame:
    """Read every station CSV from the nested station archive.

    The outer UCI archive also contains unrelated files (``data.csv``,
    ``test.csv``, and an image); only ``STATION_ARCHIVE`` is read.

    Args:
        archive_path: Path returned by :func:`download_air_quality`.

    Returns:
        Raw rows with the original columns, concatenated in station-file name
        order and in file row order within each station. No values are changed.
    """
    with zipfile.ZipFile(archive_path) as outer:
        inner_bytes = io.BytesIO(outer.read(STATION_ARCHIVE))
    with zipfile.ZipFile(inner_bytes) as inner:
        names = sorted(name for name in inner.namelist() if name.endswith(".csv"))
        frames = [pd.read_csv(inner.open(name)) for name in names]
    return pd.concat(frames, ignore_index=True)


def add_timestamp(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``raw`` with a ``timestamp`` column.

    The timestamp is built from the ``year``, ``month``, ``day``, and ``hour``
    columns as naive local time. Row order is unchanged so that the audit can
    detect unsorted or duplicated records.

    Raises:
        ValueError: If a row does not form a valid calendar date and hour.
    """
    result = raw.copy()
    result["timestamp"] = pd.to_datetime(raw[TIMESTAMP_PARTS])
    return result


def _longest_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    blocks = (mask != mask.shift()).cumsum()
    return int(mask.groupby(blocks).sum().max())


def audit_stations(frame: pd.DataFrame, target: str = "PM2.5") -> pd.DataFrame:
    """Summarize timestamp quality and target missingness for each station.

    Args:
        frame: Output of :func:`add_timestamp`, with a ``station`` column.
        target: Column whose missing values are counted.

    Returns:
        One row per station, indexed by station name, with columns:

        - ``rows``: records in the file.
        - ``start``, ``end``: first and last timestamp.
        - ``sorted``: whether file order is non-decreasing in time.
        - ``duplicate_timestamps``: records repeating an earlier timestamp.
        - ``missing_timestamps``: hours absent from the regular hourly grid
          between ``start`` and ``end``.
        - ``target_missing``, ``target_missing_rate``: missing target values
          among the records.
        - ``longest_target_gap_hours``: longest run of consecutive grid hours
          without a target value, counting absent hours as missing. For
          duplicated hours, the first record is used.
    """
    rows = {}
    for station, group in frame.groupby("station", sort=True):
        timestamps = group["timestamp"]
        first = ~timestamps.duplicated()
        grid = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
        on_grid = group.loc[first].set_index("timestamp")[target].reindex(grid)
        missing = group[target].isna()
        rows[station] = {
            "rows": len(group),
            "start": timestamps.min(),
            "end": timestamps.max(),
            "sorted": timestamps.is_monotonic_increasing,
            "duplicate_timestamps": int((~first).sum()),
            "missing_timestamps": len(grid.difference(timestamps)),
            "target_missing": int(missing.sum()),
            "target_missing_rate": float(missing.mean()),
            "longest_target_gap_hours": _longest_run(on_grid.isna()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("station")


def chronological_split(data, *, validation_start, test_start):
    """Split time-indexed data into consecutive train, validation, and test parts.

    Intervals are half-open: train is ``[first, validation_start)``, validation
    is ``[validation_start, test_start)``, and test is ``[test_start, last]``.
    Passing equal boundaries gives an empty validation part.

    Args:
        data: Series or DataFrame with a strictly increasing ``DatetimeIndex``.
        validation_start, test_start: Boundaries accepted by ``pd.Timestamp``.

    Returns:
        ``(train, validation, test)`` slices of ``data``.

    Raises:
        TypeError: If the index is not a ``DatetimeIndex``.
        ValueError: If the index is not strictly increasing, the boundaries are
            out of order, or the train or test part is empty.
    """
    index = data.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("index must be strictly increasing; sort and deduplicate first")
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if test_start < validation_start:
        raise ValueError("validation_start must not be later than test_start")

    train = data.loc[index < validation_start]
    validation = data.loc[(index >= validation_start) & (index < test_start)]
    test = data.loc[index >= test_start]
    for name, part in (("train", train), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} interval is empty")
    return train, validation, test
