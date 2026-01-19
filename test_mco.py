#!/usr/bin/env python3
"""
Unit tests for MCO pipeline components.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from mco.labels import _mco_for_horizon_series
from mco.data_validation import validate_customer_daily, validate_labels, validate_features


class TestMCOCalculation(unittest.TestCase):
    """Test MCO label computation."""

    def test_mco_simple_positive_flow(self):
        """Test MCO with constant positive outflow."""
        netflow = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        result = _mco_for_horizon_series(netflow, H=2)
        # Cumsum: [10, 20, 20, 20], max per position for H=2: [20, 20, NaN, NaN]
        self.assertAlmostEqual(result[0], 20.0, places=2)
        self.assertAlmostEqual(result[1], 20.0, places=2)
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))

    def test_mco_mixed_flow(self):
        """Test MCO with mixed inflow/outflow."""
        netflow = np.array([10.0, -5.0, 15.0, -8.0], dtype=np.float64)
        result = _mco_for_horizon_series(netflow, H=2)
        # Future windows from t+1: [[-5, 15], [15, -8]]
        # Cumsums: [[−5, 10], [15, 7]]
        # Maxes: [10, 15]
        self.assertAlmostEqual(result[0], 10.0, places=2)
        self.assertAlmostEqual(result[1], 15.0, places=2)
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))

    def test_mco_horizon_1(self):
        """Test MCO with H=1 (next day only)."""
        netflow = np.array([5.0, 10.0, 3.0], dtype=np.float64)
        result = _mco_for_horizon_series(netflow, H=1)
        # Future: [10, 3, NaN]
        self.assertAlmostEqual(result[0], 10.0, places=2)
        self.assertAlmostEqual(result[1], 3.0, places=2)
        self.assertTrue(np.isnan(result[2]))

    def test_mco_insufficient_future(self):
        """Test MCO when H exceeds data length."""
        netflow = np.array([1.0, 2.0], dtype=np.float64)
        result = _mco_for_horizon_series(netflow, H=5)
        # All NaN since horizon > length
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))


class TestDataValidation(unittest.TestCase):
    """Test data validation utilities."""

    def test_validate_customer_daily_valid(self):
        """Test valid customer_daily passes validation."""
        # Create realistic dataset with enough history
        df = pd.DataFrame({
            "customer_id": [1]*30 + [2]*30,
            "as_of_date": list(pd.date_range("2025-01-01", periods=30, freq="D")) * 2,
            "base_balance_eod": [1000.0]*60,
            "netflow_sgd": [100.0]*60,
            "inflow_sgd": [200.0]*60,
            "outflow_sgd": [100.0]*60,
        })
        result = validate_customer_daily(df, min_days=20)
        self.assertIsNone(result)

    def test_validate_customer_daily_missing_column(self):
        """Test validation fails when column is missing."""
        df = pd.DataFrame({
            "customer_id": [1, 2],
            "as_of_date": pd.date_range("2025-01-01", periods=2),
            # Missing: base_balance_eod
        })
        result = validate_customer_daily(df)
        self.assertIn("Missing columns", result)

    def test_validate_customer_daily_empty(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame({
            "customer_id": [],
            "as_of_date": [],
            "base_balance_eod": [],
            "netflow_sgd": [],
            "inflow_sgd": [],
            "outflow_sgd": [],
        })
        result = validate_customer_daily(df)
        self.assertEqual(result, "customer_daily is empty")

    def test_validate_labels_valid(self):
        """Test valid labels pass validation."""
        df = pd.DataFrame({
            "customer_id": [1, 1, 2, 2],
            "obs_date": pd.date_range("2025-01-01", periods=4, freq="D"),
            "horizon_bucket": ["M1", "M3", "M1", "M3"],
            "mco_amount_sgd": [100.0, 200.0, 150.0, 250.0],
        })
        result = validate_labels(df)
        self.assertIsNone(result)

    def test_validate_features_valid(self):
        """Test valid features pass validation."""
        df = pd.DataFrame({
            "avg_outflow_30": [100.0] * 10,
            "p95_outflow_30": [150.0] * 10,
            "avg_bal_30": [1000.0] * 10,
        })
        result = validate_features(df)
        self.assertIsNone(result)

    def test_validate_features_low_coverage(self):
        """Test validation warns on low coverage but passes if not critical."""
        df = pd.DataFrame({
            "col1": [1.0] * 10,
            "col2": [1.0] * 9 + [np.nan],  # 90% coverage
        })
        result = validate_features(df, min_coverage=0.8)
        # Should pass since 90% > 80% minimum and no critical (<50%) columns
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
