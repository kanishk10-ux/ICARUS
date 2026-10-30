"""
Comprehensive Test Suite for ICARUS Common Scientific Data Ingestion Framework
==============================================================================
Tests file format inspection, loading, and centralized dataset registry
across all 8 supported scientific domains:
  1. ScientificDataReader (icarus.data.reader)
  2. ScientificDataRegistry (icarus.data.registry)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from icarus.data.reader import ScientificDataReader
from icarus.data.registry import ScientificDataRegistry
from icarus.schemas.base import DomainEnum


# =====================================================================
# 1. ScientificDataReader Tests
# =====================================================================

def test_reader_format_detection_and_supported():
    assert ".tif" in ScientificDataReader.SUPPORTED_EXTENSIONS
    assert ".nc" in ScientificDataReader.SUPPORTED_EXTENSIONS
    assert ".npy" in ScientificDataReader.SUPPORTED_EXTENSIONS
    assert ".csv" in ScientificDataReader.SUPPORTED_EXTENSIONS
    assert ".fits" in ScientificDataReader.SUPPORTED_EXTENSIONS

    assert ScientificDataReader.detect_format("sample.tif") == "geotiff"
    assert ScientificDataReader.detect_format("sample.tiff") == "geotiff"
    assert ScientificDataReader.detect_format("observation.nc") == "netcdf"
    assert ScientificDataReader.detect_format("data.h5") == "hdf5"
    assert ScientificDataReader.detect_format("galaxy.fits") == "fits"
    assert ScientificDataReader.detect_format("grid.npy") == "numpy"
    assert ScientificDataReader.detect_format("measurements.csv") == "csv"


def test_reader_numpy_and_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Test .npy reading
        arr = np.random.uniform(0, 100, (10, 10)).astype(np.float32)
        npy_file = tmp_path / "test_grid.npy"
        np.save(str(npy_file), arr)

        inspect_res = ScientificDataReader.inspect(npy_file)
        assert inspect_res["format"] == "numpy"
        assert inspect_res["shape"] == [10, 10]
        assert "float32" in str(inspect_res["dtype"])

        loaded_arr, meta = ScientificDataReader.load(npy_file)
        assert np.array_equal(arr, loaded_arr)
        assert meta["format"] == "numpy"

        # 2. Test .csv reading
        csv_file = tmp_path / "test_data.csv"
        csv_content = "time,temperature,salinity\n1.0,24.5,35.2\n2.0,24.8,35.1\n3.0,25.0,35.0\n"
        csv_file.write_text(csv_content, encoding="utf-8")

        inspect_csv = ScientificDataReader.inspect(csv_file)
        assert inspect_csv["format"] == "csv"
        assert "columns" in inspect_csv

        csv_arr, csv_meta = ScientificDataReader.load(csv_file)
        assert csv_arr.shape == (3, 3)
        assert csv_meta["format"] == "csv"


def test_reader_missing_file_handling():
    res = ScientificDataReader.inspect("non_existent_file_xyz123.nc")
    assert "error" in res

    with pytest.raises(FileNotFoundError):
        ScientificDataReader.load("non_existent_file_xyz123.nc")


# =====================================================================
# 2. ScientificDataRegistry Tests
# =====================================================================

def test_data_registry_domain_coverage():
    registry = ScientificDataRegistry()
    all_datasets = registry.list_datasets()
    assert len(all_datasets) >= 9

    expected_domains = [
        DomainEnum.EARTH_OBSERVATION,
        DomainEnum.OCEANS,
        DomainEnum.ATMOSPHERE,
        DomainEnum.SOLID_EARTH,
        DomainEnum.CLIMATE,
        DomainEnum.HELIOPHYSICS,
        DomainEnum.PLANETARY,
        DomainEnum.ASTRONOMY,
    ]

    for dom in expected_domains:
        datasets = registry.list_datasets(domain=dom)
        assert len(datasets) > 0, f"No datasets registered for domain: {dom}"


def test_data_registry_authoritative_providers():
    registry = ScientificDataRegistry()

    # Earth Observation: NASA HLS
    hls = registry.get_dataset("nasa_hls_v2")
    assert hls is not None
    assert hls.domain == DomainEnum.EARTH_OBSERVATION
    assert "NASA" in hls.provider

    # Solid Earth: IRIS EarthScope
    iris = registry.get_dataset("iris_fdsn_seismic")
    assert iris is not None
    assert iris.domain == DomainEnum.SOLID_EARTH

    # Heliophysics: NASA SDO AIA
    sdo = registry.get_dataset("nasa_sdo_aia")
    assert sdo is not None
    assert sdo.domain == DomainEnum.HELIOPHYSICS

    # Planetary Science: NASA LRO LROC
    lroc = registry.get_dataset("nasa_lro_lroc")
    assert lroc is not None
    assert lroc.domain == DomainEnum.PLANETARY

    # Astronomy: SDSS DR18
    sdss = registry.get_dataset("sdss_dr18")
    assert sdss is not None
    assert sdss.domain == DomainEnum.ASTRONOMY
