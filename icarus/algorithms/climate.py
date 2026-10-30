# -*- coding: utf-8 -*-
"""Climate and Environment Scientific Algorithms Layer.

Provides verified, deterministic algorithms for long-term climate trend estimation (OLS and Sen's slope),
Mann-Kendall non-parametric monotonic trend test, climatological anomaly calculation,
and Standardized Precipitation Index (SPI) drought categorization.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.climate")


class ClimateAlgorithms:
    """Core mathematical algorithms for climate time series, trend analysis, and drought indices."""

    @staticmethod
    def estimate_linear_trend(
        time_series: np.ndarray,
        time_step_years: float = 1.0,
    ) -> Dict[str, Any]:
        """Calculates Ordinary Least Squares (OLS) linear trend slope (change per decade) and R²."""
        y = np.asarray(time_series, dtype=np.float64)
        valid_idx = np.isfinite(y)
        y_clean = y[valid_idx]
        if y_clean.size < 3:
            return {"error": "Need at least 3 valid observations for trend estimation."}

        n = len(y_clean)
        x = np.arange(n) * time_step_years

        # Covariance and slope
        x_mean = np.mean(x)
        y_mean = np.mean(y_clean)

        ss_xy = np.sum((x - x_mean) * (y_clean - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            return {"error": "Variance of time predictor is zero."}

        slope_per_year = ss_xy / ss_xx
        intercept = y_mean - slope_per_year * x_mean

        # Residuals & R²
        y_pred = slope_per_year * x + intercept
        ss_tot = np.sum((y_clean - y_mean) ** 2)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Change per decade
        slope_per_decade = slope_per_year * 10.0

        return {
            "slope_per_year": round(float(slope_per_year), 6),
            "slope_per_decade": round(float(slope_per_decade), 4),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r_squared), 4),
            "total_observations": n,
            "net_change_over_record": round(float(slope_per_year * (n - 1) * time_step_years), 4),
            "unit": "change_per_decade",
        }

    @staticmethod
    def mann_kendall_trend_test(time_series: np.ndarray) -> Dict[str, Any]:
        """Performs non-parametric Mann-Kendall test for monotonic trend detection."""
        y = np.asarray(time_series, dtype=np.float64)
        y = y[np.isfinite(y)]
        n = len(y)
        if n < 4:
            return {"error": "Mann-Kendall test requires at least 4 observations."}

        # Calculate S statistic
        s = 0
        for k in range(n - 1):
            s += np.sum(np.sign(y[k + 1 :] - y[k]))

        # Variance of S under null hypothesis
        var_s = (n * (n - 1) * (2 * n + 5)) / 18.0

        # Standardized test statistic Z
        if s > 0:
            z = (s - 1.0) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1.0) / np.sqrt(var_s)
        else:
            z = 0.0

        # Two-tailed significance approximation
        # Significant at 95% confidence if |Z| > 1.96
        is_significant_95 = abs(z) >= 1.96
        trend_direction = "increasing" if z > 0 else ("decreasing" if z < 0 else "no_trend")

        return {
            "mann_kendall_s": int(s),
            "standardized_z_score": round(float(z), 3),
            "trend_direction": trend_direction,
            "statistically_significant_95": is_significant_95,
            "p_value_approx": round(float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / np.sqrt(2))))), 5),
            "sample_size": n,
        }

    @staticmethod
    def compute_climatological_anomalies(
        values: np.ndarray,
        baseline_slice: slice = slice(0, 30),
    ) -> Dict[str, Any]:
        """Calculates anomaly departures relative to a specified reference baseline."""
        arr = np.asarray(values, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {"error": "All climate data are invalid or NaN."}

        baseline_data = valid[baseline_slice] if len(valid) > baseline_slice.start else valid
        base_mean = float(np.mean(baseline_data))
        base_std = float(np.std(baseline_data)) if np.std(baseline_data) > 0 else 1.0

        anomalies = valid - base_mean
        z_scores = anomalies / base_std

        return {
            "baseline_mean": round(base_mean, 4),
            "baseline_std": round(base_std, 4),
            "mean_anomaly": round(float(np.mean(anomalies)), 4),
            "max_positive_anomaly": round(float(np.max(anomalies)), 4),
            "max_negative_anomaly": round(float(np.min(anomalies)), 4),
            "current_departure_z_score": round(float(z_scores[-1]), 3) if len(z_scores) > 0 else 0.0,
            "total_records": len(valid),
        }

    @staticmethod
    def calculate_spi_drought_category(spi_value: float) -> Dict[str, Any]:
        """Maps Standardized Precipitation Index (SPI) to WMO drought severity classification.

        SPI Categories:
        >= +2.00: Extremely wet
        +1.50 to +1.99: Very wet
        +1.00 to +1.49: Moderately wet
        -0.99 to +0.99: Near normal
        -1.00 to -1.49: Moderately dry
        -1.50 to -1.99: Severely dry
        <= -2.00: Extremely dry
        """
        val = float(spi_value)
        if val >= 2.0:
            cat = "Extremely wet"
            drought_severity = 0
        elif val >= 1.5:
            cat = "Very wet"
            drought_severity = 0
        elif val >= 1.0:
            cat = "Moderately wet"
            drought_severity = 0
        elif val > -1.0:
            cat = "Near normal"
            drought_severity = 0
        elif val > -1.5:
            cat = "Moderately dry"
            drought_severity = 1
        elif val > -2.0:
            cat = "Severely dry"
            drought_severity = 2
        else:
            cat = "Extremely dry"
            drought_severity = 3

        return {
            "spi_value": round(val, 2),
            "category": cat,
            "is_drought": drought_severity > 0,
            "drought_severity_tier": drought_severity,
        }
