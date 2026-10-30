# -*- coding: utf-8 -*-
"""Centralized Dataset and Data-Provider Registry for ICARUS.

Catalogues authoritative scientific data sources across all eight domains, documenting
official access methods, authentication requirements, spatial/temporal coverage,
supported formats, licenses, and scientific limitations.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from icarus.schemas.base import DomainEnum


class DatasetRecord(BaseModel):
    """Metadata specification for an authoritative scientific dataset source."""
    id: str
    name: str
    domain: DomainEnum
    provider: str
    access_method: str = Field(..., description="e.g. STAC API, OPeNDAP, FDSN Web Service, HTTP, S3")
    endpoint_url: str
    supported_formats: List[str]
    spatial_coverage: str
    temporal_coverage: str
    resolution: str
    authentication_required: bool
    auth_provider: Optional[str] = None
    license: str
    known_limitations: List[str]


class ScientificDataRegistry:
    """Central registry providing verified data sources for all eight scientific domains."""

    def __init__(self):
        self._datasets: Dict[str, DatasetRecord] = {}
        self._initialize_core_registry()

    def _initialize_core_registry(self):
        datasets = [
            # 1. Earth Observation
            DatasetRecord(
                id="nasa_hls_v2",
                name="NASA Harmonized Landsat and Sentinel-2 (HLS) v2.0",
                domain=DomainEnum.EARTH_OBSERVATION,
                provider="NASA LP DAAC / USGS",
                access_method="STAC API / Cloud-Optimized GeoTIFF (AWS S3)",
                endpoint_url="https://cmr.earthdata.nasa.gov/stac/LPCLOUD",
                supported_formats=["geotiff", "cog"],
                spatial_coverage="Global landmass (83°N to 83°S)",
                temporal_coverage="2013-present (2-3 day revisit)",
                resolution="30m spatial resolution",
                authentication_required=True,
                auth_provider="NASA Earthdata Login (netrc / OAuth)",
                license="NASA Open Data Policy (Public Domain)",
                known_limitations=[
                    "Cloud contamination in tropical and high-latitude zones",
                    "BRDF normalization uses fixed viewing angle assumption",
                ],
            ),
            DatasetRecord(
                id="copernicus_sentinel2_l2a",
                name="Copernicus Sentinel-2 MSI Level-2A Surface Reflectance",
                domain=DomainEnum.EARTH_OBSERVATION,
                provider="European Space Agency (ESA)",
                access_method="OData API / CDSE STAC",
                endpoint_url="https://catalogue.dataspace.copernicus.eu/stac",
                supported_formats=["safe", "geotiff", "jp2"],
                spatial_coverage="Global land and coastal waters (56°S to 84°N)",
                temporal_coverage="2015-present (5-day revisit with 2A/2B constellation)",
                resolution="10m, 20m, 60m depending on spectral band",
                authentication_required=True,
                auth_provider="Copernicus Data Space Ecosystem (CDSE) Keycloak OAuth",
                license="Creative Commons CC-BY 4.0 / Copernicus Open Access",
                known_limitations=[
                    "Atmospheric correction (Sen2Cor) can have artifacts over snow and water surfaces",
                ],
            ),

            # 2. Oceans
            DatasetRecord(
                id="noaa_oisst_v2_1",
                name="NOAA Daily Optimum Interpolation Sea Surface Temperature (OISST) v2.1",
                domain=DomainEnum.OCEANS,
                provider="NOAA National Centers for Environmental Information (NCEI)",
                access_method="OPeNDAP / HTTPS / THREDDS",
                endpoint_url="https://www.ncei.noaa.gov/thredds/dodsC/OISST-V2-DAILY",
                supported_formats=["netcdf"],
                spatial_coverage="Global ocean (0° to 360°, 90°S to 90°N)",
                temporal_coverage="1981-present (Daily)",
                resolution="0.25° x 0.25° grid (~25 km)",
                authentication_required=False,
                license="US Government Public Domain",
                known_limitations=[
                    "Interpolated in coastal zones (<25 km from shore) with higher uncertainty",
                    "Does not resolve sub-mesoscale ocean eddies",
                ],
            ),

            # 3. Atmosphere
            DatasetRecord(
                id="ecmwf_era5_hourly",
                name="ECMWF ERA5 Atmospheric Reanalysis",
                domain=DomainEnum.ATMOSPHERE,
                provider="Copernicus Climate Change Service (C3S) / ECMWF",
                access_method="CDS API (Python cdsapi package)",
                endpoint_url="https://cds.climate.copernicus.eu/api",
                supported_formats=["netcdf", "grib"],
                spatial_coverage="Global atmosphere (0.25° surface, 137 vertical levels)",
                temporal_coverage="1940-present (Hourly)",
                resolution="31 km horizontal grid (~0.25°)",
                authentication_required=True,
                auth_provider="Copernicus Climate Data Store (CDS) API key",
                license="Copernicus License (Free open access for commercial and academic use)",
                known_limitations=[
                    "5-day latency for real-time operations",
                    "Convective precipitation parametrized rather than explicitly resolved",
                ],
            ),

            # 4. Solid Earth
            DatasetRecord(
                id="iris_fdsn_seismic",
                name="EarthScope / IRIS DMC Global Seismic Networks",
                domain=DomainEnum.SOLID_EARTH,
                provider="EarthScope Consortium (formerly IRIS / USGS)",
                access_method="FDSN Web Services (dataselect / event / station via ObsPy)",
                endpoint_url="https://service.iris.edu/fdsnws",
                supported_formats=["miniseed", "sac"],
                spatial_coverage="Global seismic station network",
                temporal_coverage="1970-present (Continuous and triggered waveforms)",
                resolution="Typically 20 Hz to 100 Hz sampling rates",
                authentication_required=False,
                license="Open Data Policy (Unrestricted access)",
                known_limitations=[
                    "Uneven station distribution across oceanic basins and southern hemisphere",
                    "Instrument clock drift and telemetry drops in remote stations",
                ],
            ),

            # 5. Climate
            DatasetRecord(
                id="nasa_gistemp_v4",
                name="NASA GISS Surface Temperature Analysis (GISTEMP v4)",
                domain=DomainEnum.CLIMATE,
                provider="NASA Goddard Institute for Space Studies",
                access_method="HTTPS / CSV Table & Gridded NetCDF",
                endpoint_url="https://data.giss.nasa.gov/gistemp",
                supported_formats=["netcdf", "csv"],
                spatial_coverage="Global land-ocean surface (2° x 2° and 1200 km smoothing)",
                temporal_coverage="1880-present (Monthly)",
                resolution="2° x 2° surface grid",
                authentication_required=False,
                license="US Government Public Domain",
                known_limitations=[
                    "Spatial coverage sparser in pre-1900 record and polar regions",
                    "Urban heat island effect adjustments involve statistical models",
                ],
            ),

            # 6. Heliophysics
            DatasetRecord(
                id="nasa_sdo_aia",
                name="NASA Solar Dynamics Observatory (SDO) Atmospheric Imaging Assembly (AIA)",
                domain=DomainEnum.HELIOPHYSICS,
                provider="Joint Science Operations Center (JSOC) / Stanford & NASA",
                access_method="VSO (Virtual Solar Observatory) / JSOC Web API via SunPy",
                endpoint_url="http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch",
                supported_formats=["fits"],
                spatial_coverage="Full solar disk (4096 x 4096 pixels)",
                temporal_coverage="2010-present (12-second cadence for EUV)",
                resolution="0.6 arcsec/pixel (4k x 4k)",
                authentication_required=True,
                auth_provider="JSOC registered email address",
                license="NASA Open Science Policy",
                known_limitations=[
                    "Large data volume (TB/day); requires spatial cutout requests",
                    "EUV degradation over time requires periodic calibration adjustments",
                ],
            ),

            # 7. Planetary Science
            DatasetRecord(
                id="nasa_lro_lroc",
                name="Lunar Reconnaissance Orbiter Camera (LROC) NAC / WAC Archives",
                domain=DomainEnum.PLANETARY,
                provider="NASA Planetary Data System (PDS) / Arizona State University",
                access_method="PDS Cartography and Imaging Sciences Node (Ciss / STAC API)",
                endpoint_url="https://pds-imaging.jpl.nasa.gov/service/search",
                supported_formats=["pds3", "pds4", "geotiff"],
                spatial_coverage="Entire lunar surface (0.5m/pixel for NAC targeted, 100m for WAC)",
                temporal_coverage="2009-present",
                resolution="Down to 0.5 meters per pixel (LROC NAC)",
                authentication_required=False,
                license="NASA Public Domain",
                known_limitations=[
                    "Incidence angle and illumination vary heavily with lunar phase",
                    "Polar shadowed regions (PSR) require secondary scattered light imaging",
                ],
            ),

            # 8. Astronomy & Astrophysics
            DatasetRecord(
                id="sdss_dr18",
                name="Sloan Digital Sky Survey Data Release 18 (SDSS DR18)",
                domain=DomainEnum.ASTRONOMY,
                provider="Astrophysical Research Consortium (ARC) & SDSS-IV/V",
                access_method="CasJobs / SkyServer SQL API & FITS Cutout Service",
                endpoint_url="https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
                supported_formats=["fits", "csv"],
                spatial_coverage="Northern & Southern Galactic Caps (~14,555 square degrees)",
                temporal_coverage="2000-present",
                resolution="0.396 arcsec/pixel in 5 optical bands (u, g, r, i, z)",
                authentication_required=False,
                license="SDSS Open Access (Attribution required in publications)",
                known_limitations=[
                    "Bright star saturation limit at r ~ 14 mag",
                    "Extinction corrections near Galactic plane introduce systematic uncertainty",
                ],
            ),
        ]

        for ds in datasets:
            self._datasets[ds.id] = ds

    def list_datasets(self, domain: Optional[DomainEnum] = None) -> List[DatasetRecord]:
        """Returns registered scientific datasets, optionally filtered by domain."""
        if domain is None:
            return list(self._datasets.values())
        return [ds for ds in self._datasets.values() if ds.domain == domain]

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Retrieves single dataset record by identifier."""
        return self._datasets.get(dataset_id)
