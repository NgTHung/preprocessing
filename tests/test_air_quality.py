import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from preprocessing.air_quality import (
    STATION_ARCHIVE,
    add_timestamp,
    audit_stations,
    chronological_split,
    download_air_quality,
    read_stations,
)

COLUMNS = ["No", "year", "month", "day", "hour", "PM2.5", "station"]


def station_csv(station, rows):
    frame = pd.DataFrame(
        [(i + 1, *row, station) for i, row in enumerate(rows)], columns=COLUMNS
    )
    return frame.to_csv(index=False)


def write_uci_archive(path, csv_by_station):
    """Mimic the UCI layout: an outer zip with the station zip and unrelated files."""
    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, "w") as inner:
        inner.writestr("PRSA_Data/", "")
        for station, text in csv_by_station.items():
            inner.writestr(f"PRSA_Data/PRSA_Data_{station}.csv", text)
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr(STATION_ARCHIVE, inner_bytes.getvalue())
        outer.writestr("data.csv", "unrelated,file\n1,2\n")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_keeps_verified_file(tmp_path):
    source = write_uci_archive(tmp_path / "source.zip", {"A": station_csv("A", [])})

    archive = download_air_quality(
        tmp_path / "raw", url=source.as_uri(), sha256=sha256(source)
    )

    assert archive.read_bytes() == source.read_bytes()


def test_download_rejects_checksum_mismatch(tmp_path):
    source = write_uci_archive(tmp_path / "source.zip", {"A": station_csv("A", [])})

    with pytest.raises(ValueError, match="SHA-256"):
        download_air_quality(tmp_path / "raw", url=source.as_uri(), sha256="0" * 64)

    assert not any((tmp_path / "raw").rglob("*.zip"))


def test_download_reuses_existing_archive_without_network(tmp_path):
    source = write_uci_archive(tmp_path / "source.zip", {"A": station_csv("A", [])})
    archive = download_air_quality(
        tmp_path / "raw", url=source.as_uri(), sha256=sha256(source)
    )

    reused = download_air_quality(
        tmp_path / "raw", url="file:///missing.zip", sha256=sha256(source)
    )

    assert reused == archive


def test_read_stations_reads_only_station_files(tmp_path):
    archive = write_uci_archive(
        tmp_path / "uci.zip",
        {
            "B": station_csv("B", [(2013, 3, 1, 0, 5.0)]),
            "A": station_csv("A", [(2013, 3, 1, 0, 1.0), (2013, 3, 1, 1, 2.0)]),
        },
    )

    raw = read_stations(archive)

    assert list(raw.columns) == COLUMNS
    assert raw["station"].tolist() == ["A", "A", "B"]


def test_add_timestamp_builds_hours_without_reordering():
    raw = pd.DataFrame(
        {"year": [2013, 2013], "month": [3, 2], "day": [1, 28], "hour": [0, 23]}
    )

    result = add_timestamp(raw)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2013-03-01 00:00"),
        pd.Timestamp("2013-02-28 23:00"),
    ]
    assert "timestamp" not in raw.columns


def test_audit_counts_duplicates_missing_hours_and_target_gaps():
    rows = [
        (2013, 3, 1, 0, 1.0),
        (2013, 3, 1, 1, np.nan),
        (2013, 3, 1, 1, np.nan),  # duplicate hour
        # 02:00 absent from the file
        (2013, 3, 1, 3, np.nan),
        (2013, 3, 1, 4, 4.0),
    ]
    frame = add_timestamp(pd.read_csv(io.StringIO(station_csv("A", rows))))

    audit = audit_stations(frame, target="PM2.5").loc["A"]

    assert audit["rows"] == 5
    assert audit["duplicate_timestamps"] == 1
    assert audit["missing_timestamps"] == 1
    assert audit["target_missing"] == 3
    assert audit["target_missing_rate"] == pytest.approx(3 / 5)
    # 01:00, the absent 02:00, and 03:00 form one gap on the hourly grid.
    assert audit["longest_target_gap_hours"] == 3
    assert bool(audit["sorted"]) is True


def test_audit_flags_unsorted_station_file():
    rows = [(2013, 3, 1, 1, 1.0), (2013, 3, 1, 0, 2.0)]
    frame = add_timestamp(pd.read_csv(io.StringIO(station_csv("A", rows))))

    audit = audit_stations(frame, target="PM2.5").loc["A"]

    assert bool(audit["sorted"]) is False
    assert audit["longest_target_gap_hours"] == 0


def hourly(start, periods):
    index = pd.date_range(start, periods=periods, freq="h")
    return pd.Series(np.arange(periods, dtype=float), index=index)


def test_split_separates_intervals_at_boundaries():
    series = hourly("2016-01-01 00:00", 6)

    train, validation, test = chronological_split(
        series, validation_start="2016-01-01 02:00", test_start="2016-01-01 04:00"
    )

    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()
    assert len(train) + len(validation) + len(test) == len(series)
    assert test.index.min() == pd.Timestamp("2016-01-01 04:00")


def test_split_allows_empty_validation_interval():
    series = hourly("2016-01-01 00:00", 4)

    train, validation, test = chronological_split(
        series, validation_start="2016-01-01 02:00", test_start="2016-01-01 02:00"
    )

    assert (len(train), len(validation), len(test)) == (2, 0, 2)


def test_split_rejects_unsorted_index():
    series = hourly("2016-01-01 00:00", 4).iloc[[1, 0, 2, 3]]

    with pytest.raises(ValueError, match="increasing"):
        chronological_split(
            series, validation_start="2016-01-01 02:00", test_start="2016-01-01 03:00"
        )


def test_split_rejects_test_before_validation():
    series = hourly("2016-01-01 00:00", 4)

    with pytest.raises(ValueError, match="validation_start"):
        chronological_split(
            series, validation_start="2016-01-01 03:00", test_start="2016-01-01 02:00"
        )


def test_split_rejects_empty_test_interval():
    series = hourly("2016-01-01 00:00", 4)

    with pytest.raises(ValueError, match="empty"):
        chronological_split(
            series, validation_start="2016-01-01 02:00", test_start="2017-01-01"
        )
