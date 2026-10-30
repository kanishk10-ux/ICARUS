"""
Comprehensive Test Suite for ICARUS Domain Scientific Computation Framework
=============================================================================
Tests deterministic scientific computation across all 8 supported scientific domains:
  1. Earth Observation (icarus.algorithms.eo)
  2. Oceans and Water (icarus.algorithms.oceans)
  3. Atmosphere and Weather (icarus.algorithms.atmosphere)
  4. Solid Earth and Seismology (icarus.algorithms.solid_earth)
  5. Climate and Environment (icarus.algorithms.climate)
  6. Sun and Space Weather / Heliophysics (icarus.algorithms.heliophysics)
  7. Planetary Science (icarus.algorithms.planetary)
  8. Astronomy and Astrophysics (icarus.algorithms.astronomy)
"""

import numpy as np
import pytest

from icarus.algorithms.astronomy import AstronomyAlgorithms
from icarus.algorithms.atmosphere import AtmosphereAlgorithms
from icarus.algorithms.climate import ClimateAlgorithms
from icarus.algorithms.eo import EarthObservationAlgorithms
from icarus.algorithms.heliophysics import HeliophysicsAlgorithms
from icarus.algorithms.oceans import OceanAlgorithms
from icarus.algorithms.planetary import PlanetaryAlgorithms
from icarus.algorithms.solid_earth import SolidEarthAlgorithms


# =====================================================================
# 1. Earth Observation Tests
# =====================================================================

def test_eo_spectral_indices():
    nir = np.array([[0.8, 0.6], [0.7, 0.9]], dtype=np.float32)
    red = np.array([[0.2, 0.4], [0.1, 0.3]], dtype=np.float32)
    green = np.array([[0.3, 0.5], [0.2, 0.4]], dtype=np.float32)
    swir = np.array([[0.1, 0.2], [0.15, 0.25]], dtype=np.float32)

    ndvi = EarthObservationAlgorithms.compute_ndvi(nir, red)
    assert ndvi.shape == (2, 2)
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)
    assert np.isclose(ndvi[0, 0], 0.6, atol=1e-4)

    ndwi = EarthObservationAlgorithms.compute_ndwi(green, nir)
    assert ndwi.shape == (2, 2)
    assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)

    nbr = EarthObservationAlgorithms.compute_nbr(nir, swir)
    assert nbr.shape == (2, 2)
    assert np.all(nbr >= -1.0) and np.all(nbr <= 1.0)


def test_eo_surface_area_and_stats():
    area_res = EarthObservationAlgorithms.calculate_surface_area_km2(
        pixel_count=100, pixel_resolution_m=30.0, crs="EPSG:32633"
    )
    assert area_res["pixel_count"] == 100
    assert area_res["area_km2"] == 0.09
    assert area_res["crs"] == "EPSG:32633"

    # Rejection of geographic coordinates
    geo_res = EarthObservationAlgorithms.calculate_surface_area_km2(
        pixel_count=100, pixel_resolution_m=30.0, crs="EPSG:4326"
    )
    assert "error" in geo_res
    assert geo_res["area_km2"] is None

    data = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    stats = EarthObservationAlgorithms.compute_spatial_statistics(data)
    assert stats["total_pixels"] == 100
    assert np.isclose(stats["mean"], 50.5)
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0


# =====================================================================
# 2. Oceans and Water Tests
# =====================================================================

def test_oceans_sst_and_heatwaves():
    current_sst = np.array([28.5, 29.0, 30.2, 27.8])
    climatological_mean = np.array([26.0, 26.5, 27.0, 26.0])

    anom = OceanAlgorithms.analyze_sst_anomalies(
        sst_observed_celsius=current_sst,
        sst_climatology_celsius=climatological_mean,
        mhw_threshold_celsius=1.5
    )
    assert "mean_anomaly_celsius" in anom
    assert np.isclose(anom["mean_anomaly_celsius"], 2.5)
    assert anom["marine_heatwave_fraction"] > 0.0


def test_oceans_oc4_chlorophyll():
    rrs_blue = np.array([[0.006, 0.005], [0.004, 0.006]])
    rrs_green = np.array([[0.002, 0.002], [0.002, 0.002]])
    chl = OceanAlgorithms.estimate_chlorophyll_oc4(rrs_blue=rrs_blue, rrs_green=rrs_green)
    assert chl.shape == (2, 2)
    assert np.all(chl > 0.0)


def test_oceans_velocity_vorticity():
    u = np.array([[0.5, 0.6], [0.5, 0.6]])
    v = np.array([[0.2, 0.2], [0.3, 0.3]])
    curr = OceanAlgorithms.compute_ocean_current_kinematics(
        u_velocity_mps=u, v_velocity_mps=v, dx_m=1000.0, dy_m=1000.0
    )
    assert curr["max_current_speed_mps"] > 0.0
    assert "mean_vorticity_per_sec" in curr


# =====================================================================
# 3. Atmosphere and Weather Tests
# =====================================================================

def test_atmosphere_wind_vectors():
    u = np.array([10.0, 0.0])
    v = np.array([0.0, 10.0])
    vectors = AtmosphereAlgorithms.compute_wind_vectors(u_mps=u, v_mps=v)
    assert "mean_wind_speed_mps" in vectors
    assert "mean_wind_direction_deg" in vectors
    assert np.isclose(vectors["mean_wind_speed_mps"], 10.0)


def test_atmosphere_precipitation_and_lapse_rate():
    precip_series = np.array([1.5, 2.0, 0.5, 5.0, 12.0, 0.0])
    acc = AtmosphereAlgorithms.analyze_precipitation(precip_mm_per_hour=precip_series, duration_hours=6.0)
    assert "mean_rainfall_rate_mmph" in acc
    assert acc["total_areal_accumulation_mm"] > 0

    temps_2d = np.array([[20.0, 21.0, 22.0], [20.5, 21.5, 22.5], [21.0, 22.0, 23.0]], dtype=np.float64)
    grad = AtmosphereAlgorithms.compute_temperature_gradient(temp_celsius=temps_2d, dx_km=10.0, dy_km=10.0)
    assert "mean_gradient_celsius_per_100km" in grad
    assert grad["mean_gradient_celsius_per_100km"] is not None


# =====================================================================
# 4. Solid Earth and Seismology Tests
# =====================================================================

def test_solid_earth_preprocessing():
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10 * np.pi, 500)) + 5.0
    processed = SolidEarthAlgorithms.preprocess_waveform(signal, taper_fraction=0.05)
    assert len(processed) == 500
    assert np.abs(np.mean(processed)) < 0.1


def test_solid_earth_snr_and_fft():
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, 500)
    signal = np.sin(np.linspace(0, 20 * np.pi, 500)) + np.random.normal(0, 0.05, 500)
    full_trace = np.concatenate([noise, signal])

    snr_db = SolidEarthAlgorithms.compute_snr_db(full_trace, noise_window_samples=500, signal_window_samples=500)
    assert snr_db > 10.0

    fft_res = SolidEarthAlgorithms.compute_frequency_spectrum(signal, sampling_rate_hz=100.0)
    assert fft_res["dominant_frequency_hz"] > 0.0


def test_solid_earth_shaking_metrics():
    accel = np.array([0.05, 0.12, -0.25, 0.18, -0.04])
    metrics = SolidEarthAlgorithms.compute_ground_motion_metrics(accel, sampling_rate_hz=100.0)
    assert np.isclose(metrics["pga_mps2"], 0.25)
    assert metrics["arias_intensity_mps"] >= 0.0


# =====================================================================
# 5. Climate and Environment Tests
# =====================================================================

def test_climate_trends():
    time_series = np.linspace(1980, 2020, 41)
    temps = 14.0 + 0.02 * (time_series - 1980) + np.random.normal(0, 0.01, 41)

    slope_res = ClimateAlgorithms.estimate_linear_trend(temps, time_step_years=1.0)
    assert np.isclose(slope_res["slope_per_decade"], 0.20, atol=0.05)
    assert slope_res["r_squared"] > 0.85

    mk_res = ClimateAlgorithms.mann_kendall_trend_test(temps)
    assert mk_res["trend_direction"] == "increasing"
    assert bool(mk_res["statistically_significant_95"]) is True


def test_climate_spi():
    spi_res = ClimateAlgorithms.calculate_spi_drought_category(-2.15)
    assert spi_res["category"] == "Extremely dry"
    assert spi_res["is_drought"] is True
    assert spi_res["drought_severity_tier"] == 3


# =====================================================================
# 6. Heliophysics and Space Weather Tests
# =====================================================================

def test_heliophysics_flare_and_storms():
    # X2.5 Flare
    x_flare = HeliophysicsAlgorithms.classify_goes_solar_flare(2.5e-4)
    assert x_flare["goes_flare_class"] == "X2.5"
    assert x_flare["severity_index"] == 4

    # C1.0 Flare
    c_flare = HeliophysicsAlgorithms.classify_goes_solar_flare(1.0e-6)
    assert c_flare["goes_flare_class"] == "C1.0"
    assert c_flare["severity_index"] == 2

    # Severe Storm (G4)
    storm = HeliophysicsAlgorithms.evaluate_geomagnetic_storm(kp_index=8.0, dst_index_nt=-220.0)
    assert "G4" in storm["noaa_g_scale"]
    assert storm["dst_storm_status"] in ("Super-storm", "Intense Storm")


def test_heliophysics_differential_rotation():
    equator = HeliophysicsAlgorithms.compute_solar_differential_rotation(latitude_deg=0.0)
    pole = HeliophysicsAlgorithms.compute_solar_differential_rotation(latitude_deg=75.0)
    assert equator["angular_velocity_deg_per_day"] > pole["angular_velocity_deg_per_day"]
    assert equator["synodic_rotation_period_days"] < pole["synodic_rotation_period_days"]


# =====================================================================
# 7. Planetary Science Tests
# =====================================================================

def test_planetary_topography():
    x = np.arange(50) * 10.0
    y = np.arange(50) * 10.0
    xx, yy = np.meshgrid(x, y)
    tilted_plane = (0.5 * xx).astype(np.float32)

    topo = PlanetaryAlgorithms.compute_terrain_slope_and_aspect(elevation_dem_m=tilted_plane, pixel_resolution_m=10.0)
    assert np.isclose(topo["mean_slope_degrees"], 26.56, atol=1.0)
    assert topo["terrain_roughness_index"] >= 0.0


def test_planetary_crater_morphometry():
    morph = PlanetaryAlgorithms.analyze_crater_morphometry(
        diameter_km=0.8, depth_km=0.16
    )
    assert "Simple crater" in morph["morphology_class"]
    assert np.isclose(morph["depth_to_diameter_ratio"], 0.2, atol=1e-3)


# =====================================================================
# 8. Astronomy and Astrophysics Tests
# =====================================================================

def test_astronomy_light_curve():
    np.random.seed(42)
    true_period = 2.5
    time_days = np.linspace(0, 25, 250)
    flux = 100.0 + 5.0 * np.sin(2 * np.pi * time_days / true_period) + np.random.normal(0, 0.5, 250)

    res = AstronomyAlgorithms.analyze_light_curve(
        times_days=time_days, flux=flux, min_period_days=1.0, max_period_days=5.0
    )
    assert "detected_period_days" in res
    assert np.isclose(res["detected_period_days"], true_period, atol=0.1)


def test_astronomy_aperture_photometry_and_asymmetry():
    img = np.zeros((51, 51), dtype=np.float32)
    img[25, 25] = 1000.0

    phot = AstronomyAlgorithms.compute_aperture_photometry(
        image=img, center_x=25, center_y=25, aperture_radius_pixels=3.0, background_inner_radius=5.0, background_outer_radius=10.0
    )
    assert phot["net_source_flux"] > 900.0
    assert phot["instrumental_magnitude"] < 20.0

    y, x = np.ogrid[:41, :41]
    symmetric_galaxy = np.exp(-((x - 20)**2 + (y - 20)**2) / 30.0).astype(np.float32)
    asym = AstronomyAlgorithms.compute_galaxy_cas_asymmetry(symmetric_galaxy)
    assert asym["asymmetry_index_a"] < 0.05
    assert asym["morphology_indicator"] == "regular_symmetric"
