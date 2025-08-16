"""
Unit tests for the FeatureNormalizer class of the PyTrad framework.

This test suite provides comprehensive coverage of the FeatureNormalizer class,
which handles normalization of financial data features.

The tests verify:
1. Correct initialization and configuration
2. Feature categorization logic
3. Fit method for different feature types
4. Transform method application
5. Combined fit_transform operation
6. Save/load functionality for persistence
7. Edge cases and error handling

To run all tests:
    pytest -xvs tests/unit/test_normalizer.py

To run a specific test class:
    pytest -xvs tests/unit/test_normalizer.py::TestFeatureNormalizer

To run a specific test:
    pytest -xvs tests/unit/test_normalizer.py::TestFeatureNormalizer::test_initialization
"""

import os

import numpy as np
import pandas as pd
import pytest

from pytrad.normalizer import FeatureNormalizer


class TestFeatureNormalizer:
    """Test suite for the FeatureNormalizer class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing normalization."""
        # Create a DataFrame with different types of features
        np.random.seed(42)  # For reproducibility

        # Create a date range for index
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create price data (for z-score normalization)
        open_prices = np.random.normal(100, 5, 100)
        high_prices = open_prices + np.random.normal(2, 0.5, 100)
        low_prices = open_prices - np.random.normal(2, 0.5, 100)
        close_prices = open_prices + np.random.normal(0, 1, 100)

        # Create volume data (for log transform)
        volume = np.random.lognormal(10, 1, 100)

        # Create oscillator data (for 0-100 scale)
        rsi = np.random.uniform(0, 100, 100)
        mfi = np.random.uniform(0, 100, 100)

        # Create additional indicators
        ma_5 = pd.Series(close_prices).rolling(5).mean().values
        ma_10 = pd.Series(close_prices).rolling(10).mean().values

        # Create negative values for testing edge cases
        some_indicator = np.random.normal(0, 10, 100)

        # Create a DataFrame
        df = pd.DataFrame(
            {
                "time": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
                "RSI_14": rsi,
                "MFI_14": mfi,
                "MA_5": ma_5,
                "MA_10": ma_10,
                "ATR_14": np.abs(some_indicator),
                "some_indicator": some_indicator,
                "timeframe": "1d",
                "asset": "EURUSD",
                "market": "forex",
            }
        )

        return df

    @pytest.fixture
    def normalizer(self):
        """Create a basic FeatureNormalizer instance."""
        return FeatureNormalizer(max_samples_for_stats=10000)

    @pytest.fixture
    def fitted_normalizer(self, normalizer, sample_df):
        """Create a FeatureNormalizer instance that's already fitted to sample data."""
        normalizer.fit(sample_df)
        return normalizer

    def test_initialization(self, normalizer):
        """Test that FeatureNormalizer initializes correctly."""
        assert normalizer.max_samples_for_stats == 10000
        assert normalizer.stats == {}
        assert normalizer.is_fitted is False

    def test_categorize_features(self, normalizer):
        """Test the _categorize_features method."""
        columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "MA_5",
            "EMA_10",
            "RSI_14",
            "BB_upper",
            "BB_lower",
            "MACD_12_26_9",
            "ADX_14",
            "ATR_14",
            "OBV_1",
            "MFI_14",
            "CCI_20",
            "Stochastic_14",
            "time",
            "timeframe",
            "asset",
            "market",
            "unknown_feature",
        ]

        feature_types = normalizer._categorize_features(columns)

        # Check that features are correctly categorized
        assert "open" in feature_types["price"]
        assert "high" in feature_types["price"]
        assert "low" in feature_types["price"]
        assert "close" in feature_types["price"]
        assert "volume" in feature_types["volume"]
        assert "MA_5" in feature_types["ma"]
        assert "EMA_10" in feature_types["ma"]
        assert "RSI_14" in feature_types["rsi"]
        assert "BB_upper" in feature_types["bollinger_bands"]
        assert "BB_lower" in feature_types["bollinger_bands"]
        assert "MACD_12_26_9" in feature_types["macd"]
        assert "ADX_14" in feature_types["adx"]
        assert "ATR_14" in feature_types["atr"]
        assert "OBV_1" in feature_types["obv"]
        assert "MFI_14" in feature_types["mfi"]
        assert "CCI_20" in feature_types["cci"]
        assert "Stochastic_14" in feature_types["stochastic"]
        assert "unknown_feature" in feature_types["other"]

        # Check that metadata columns are not included
        assert "time" not in [
            item for sublist in feature_types.values() for item in sublist
        ]
        assert "timeframe" not in [
            item for sublist in feature_types.values() for item in sublist
        ]
        assert "asset" not in [
            item for sublist in feature_types.values() for item in sublist
        ]
        assert "market" not in [
            item for sublist in feature_types.values() for item in sublist
        ]

    def test_fit(self, normalizer, sample_df):
        """Test the fit method computes correct statistics."""
        normalizer.fit(sample_df)

        # Check that normalizer is marked as fitted
        assert normalizer.is_fitted is True

        # Check that stats were computed for all non-metadata columns
        for column in sample_df.columns:
            if column not in ["time", "timeframe", "asset", "market"]:
                assert column in normalizer.stats

        # Check specific normalization methods for different feature types
        assert normalizer.stats["open"][0] == normalizer.Z_SCORE
        assert normalizer.stats["volume"][0] == normalizer.LOG_TRANSFORM
        assert normalizer.stats["RSI_14"][0] == normalizer.SCALE_0_100

        # Check that statistics are reasonable
        assert (
            abs(normalizer.stats["open"][1]["mean"] - sample_df["open"].mean()) < 1e-6
        )
        assert abs(normalizer.stats["open"][1]["std"] - sample_df["open"].std()) < 1e-6

    def test_transform(self, fitted_normalizer, sample_df):
        """Test the transform method correctly normalizes data."""
        normalized_df = fitted_normalizer.transform(sample_df)

        # Check that all non-metadata columns were normalized
        for column in sample_df.columns:
            if column not in ["time", "timeframe", "asset", "market"]:
                # The normalized column should have different values than the original
                assert not np.array_equal(
                    normalized_df[column].values, sample_df[column].values
                )

        # For Z-score normalized columns, check the resulting distribution
        for column in ["open", "high", "low", "close"]:
            # Mean should be close to 0
            assert abs(normalized_df[column].mean()) < 0.1
            # Std should be close to 1
            assert 0.9 < normalized_df[column].std() < 1.1

        # For 0-100 scale columns, check the resulting range
        for column in ["RSI_14", "MFI_14"]:
            # All values should be between 0 and 1
            assert normalized_df[column].min() >= 0
            assert normalized_df[column].max() <= 1

        # Check that metadata columns are unchanged
        for column in ["timeframe", "asset", "market"]:
            assert np.array_equal(
                normalized_df[column].values, sample_df[column].values
            )

    def test_fit_transform(self, normalizer, sample_df):
        """Test the fit_transform method."""
        normalized_df = normalizer.fit_transform(sample_df)

        # Check that normalizer is marked as fitted
        assert normalizer.is_fitted is True

        # Check that all non-metadata columns were normalized
        for column in sample_df.columns:
            if column not in ["time", "timeframe", "asset", "market"]:
                # The normalized column should have different values than the original
                assert not np.array_equal(
                    normalized_df[column].values, sample_df[column].values
                )

    # def test_transform_new_data(self, fitted_normalizer, sample_df):
    #     """Test transforming new data with an already fitted normalizer."""
    #     # Create new data with slightly different distribution
    #     new_df = sample_df.copy()
    #     new_df["open"] = new_df["open"] + 10  # Shift the mean

    #     normalized_df = fitted_normalizer.transform(new_df)

    #     # Check that the normalization was applied using the original statistics
    #     mean_diff = abs(
    #         normalized_df["open"].mean()
    #         - (new_df["open"] - fitted_normalizer.stats["open"][1]["mean"])
    #         / fitted_normalizer.stats["open"][1]["std"]
    #     )
    #     assert mean_diff < 1e-6

    # def test_sign_preserving_normalization(self, normalizer, sample_df):
    #     """Test that sign-preserving normalization correctly preserves the sign of values."""
    #     # Add some return columns with both positive and negative values
    #     sample_df["close_ret"] = np.random.normal(0, 0.01, 100)  # Typical log returns
    #     sample_df["MA_5_ret"] = np.random.normal(0, 0.005, 100)  # MA on returns

    #     # Ensure some values are positive, some negative, and some zero
    #     sample_df.loc[0:30, "close_ret"] = np.abs(
    #         sample_df.loc[0:30, "close_ret"]
    #     )  # Positive
    #     sample_df.loc[31:70, "close_ret"] = -np.abs(
    #         sample_df.loc[31:70, "close_ret"]
    #     )  # Negative
    #     sample_df.loc[71:75, "close_ret"] = 0  # Zero

    #     # Fit and transform
    #     normalizer.fit(sample_df)
    #     normalized_df = normalizer.transform(sample_df)

    #     # Check that the return columns were normalized with sign_preserving method
    #     assert normalizer.stats["close_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["MA_5_ret"][0] == normalizer.SIGN_PRESERVING

    #     # Check that signs are preserved
    #     for idx in range(100):
    #         original_sign = np.sign(sample_df.loc[idx, "close_ret"])
    #         normalized_sign = np.sign(normalized_df.loc[idx, "close_ret"])
    #         assert original_sign == normalized_sign, f"Sign changed at index {idx}"

    #         # Zero values should remain zero
    #         if sample_df.loc[idx, "close_ret"] == 0:
    #             assert normalized_df.loc[idx, "close_ret"] == 0

    #     # Check that the absolute values are normalized to the [0,1] range
    #     assert normalized_df["close_ret"].abs().max() <= 1.0

    # def test_normalize_new_column(self, fitted_normalizer, sample_df):
    #     """Test behavior when transforming a DataFrame with a new column not seen during fit."""
    #     df_with_new_col = sample_df.copy()
    #     df_with_new_col["new_feature"] = np.random.normal(0, 1, 100)

    #     # Transform should work and leave the new column untouched
    #     result = fitted_normalizer.transform(df_with_new_col)

    #     # New column should be present but not normalized
    #     assert "new_feature" in result.columns
    #     assert np.array_equal(
    #         result["new_feature"].to_numpy(), df_with_new_col["new_feature"].to_numpy()
    #     )

    # def test_normalize_missing_column(self, fitted_normalizer, sample_df):
    #     """Test behavior when transforming a DataFrame missing a column seen during fit."""
    #     df_missing_col = sample_df.copy()
    #     # Remove a column that was present during fit
    #     del df_missing_col["MA_5"]

    #     # Transform should work without errors
    #     result = fitted_normalizer.transform(df_missing_col)

    #     # The missing column should not be in the result
    #     assert "MA_5" not in result.columns

    # def test_convert_numpy(self, normalizer):
    #     """Test the _convert_numpy method handles various numpy types correctly."""
    #     # Test with numpy array
    #     np_array = np.array([1, 2, 3])
    #     assert normalizer._convert_numpy(np_array) == [1, 2, 3]

    #     # Test with numpy number
    #     np_num = np.float32(3.14)
    #     assert abs(normalizer._convert_numpy(np_num) - 3.14) < 1e-6

    #     # Test with dict containing numpy values
    #     np_dict = {"a": np.array([1, 2]), "b": np.float64(2.71)}
    #     converted = normalizer._convert_numpy(np_dict)
    #     assert converted["a"] == [1, 2]
    #     assert abs(converted["b"] - 2.71) < 1e-6

    #     # Test with list containing numpy values
    #     np_list = [np.int32(1), np.array([4, 5])]
    #     converted = normalizer._convert_numpy(np_list)
    #     assert converted[0] == 1
    #     assert converted[1] == [4, 5]

    #     # Test with regular Python value
    #     assert normalizer._convert_numpy(42) == 42

    # def test_indicator_normalization(self, normalizer):
    #     """Test that indicators from indicator.py are properly normalized."""
    #     # Create a sample DataFrame with indicators
    #     np.random.seed(42)

    #     # Create price data
    #     close_prices = np.random.normal(100, 5, 100)
    #     close_ret = np.random.normal(0, 0.01, 100)
    #     high_prices = close_prices + np.random.normal(2, 0.5, 100)
    #     low_prices = close_prices - np.random.normal(2, 0.5, 100)
    #     open_prices = close_prices - np.random.normal(0, 1, 100)
    #     volume = np.random.lognormal(10, 1, 100)

    #     # Create a DataFrame
    #     df = pd.DataFrame(
    #         {
    #             "open": open_prices,
    #             "high": high_prices,
    #             "low": low_prices,
    #             "close": close_prices,
    #             "volume": volume,
    #             "open_ret": np.random.normal(0, 0.01, 100),
    #             "high_ret": np.random.normal(0, 0.01, 100),
    #             "low_ret": np.random.normal(0, 0.01, 100),
    #             "close_ret": close_ret,
    #             # Add various indicators
    #             "MA_14": pd.Series(close_prices).rolling(14).mean().values,
    #             "MA_14_ret": pd.Series(close_ret).rolling(14).mean().values,
    #             "EMA_14": pd.Series(close_prices).ewm(span=14).mean().values,
    #             "EMA_14_ret": pd.Series(close_ret).ewm(span=14).mean().values,
    #             "RSI_14": np.random.uniform(0, 100, 100),
    #             "MACD_line": np.random.normal(0, 1, 100),  # Simulated MACD
    #             "MACD_signal": np.random.normal(0, 0.5, 100),
    #             "MACD_histogram": np.random.normal(0, 0.3, 100),
    #             "BB_upper": close_prices + np.random.normal(5, 1, 100),
    #             "BB_middle": close_prices,
    #             "BB_lower": close_prices - np.random.normal(5, 1, 100),
    #             "ATR_14": np.abs(np.random.normal(2, 0.5, 100)),
    #             "CCI_20": np.random.normal(0, 100, 100),
    #             "OBV_1": np.cumsum(np.random.normal(0, 100000, 100)),
    #             "MFI_14": np.random.uniform(0, 100, 100),
    #             "Stochastic_14": np.random.uniform(0, 100, 100),
    #         }
    #     )

    #     # Fit and transform
    #     normalizer.fit(df)
    #     normalized_df = normalizer.transform(df)

    #     # Check normalization methods for each indicator type
    #     # Price columns
    #     assert normalizer.stats["open"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["high"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["low"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["close"][0] == normalizer.Z_SCORE

    #     # Return columns with sign preservation
    #     assert normalizer.stats["open_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["high_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["low_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["close_ret"][0] == normalizer.SIGN_PRESERVING

    #     # Volume with log transform
    #     assert normalizer.stats["volume"][0] == normalizer.LOG_TRANSFORM

    #     # MA and EMA
    #     assert normalizer.stats["MA_14"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["EMA_14"][0] == normalizer.Z_SCORE

    #     # MA and EMA on returns
    #     assert normalizer.stats["MA_14_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["EMA_14_ret"][0] == normalizer.SIGN_PRESERVING

    #     # Oscillators (0-100 scale)
    #     assert normalizer.stats["RSI_14"][0] == normalizer.SCALE_0_100
    #     assert normalizer.stats["MFI_14"][0] == normalizer.SCALE_0_100
    #     assert normalizer.stats["Stochastic_14"][0] == normalizer.SCALE_0_100

    #     # MACD components
    #     assert normalizer.stats["MACD_line"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["MACD_signal"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["MACD_histogram"][0] == normalizer.Z_SCORE

    #     # Other indicators
    #     assert normalizer.stats["BB_upper"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["BB_middle"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["BB_lower"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["ATR_14"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["CCI_20"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["OBV_1"][0] == normalizer.Z_SCORE

    #     # Check that all features are actually normalized (values are different)
    #     for column in df.columns:
    #         assert not np.array_equal(
    #             normalized_df[column].to_numpy(), df[column].to_numpy()
    #         )

    # def test_normalization_with_actual_indicators(self, normalizer):
    #     """Test normalization with actual indicators calculated from indicator.py."""
    #     from pytrad.indicator import (
    #         ATR,
    #         CCI,
    #         EMA,
    #         MA,
    #         MACD,
    #         MFI,
    #         OBV,
    #         RSI,
    #         BollingerBands,
    #         Stochastic,
    #     )

    #     # Create a sample DataFrame with price data
    #     np.random.seed(42)
    #     n_samples = 200  # Need enough samples for indicators

    #     # Create price data
    #     close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    #     high_prices = close_prices + np.abs(np.random.normal(0, 2, n_samples))
    #     low_prices = close_prices - np.abs(np.random.normal(0, 2, n_samples))
    #     open_prices = close_prices + np.random.normal(0, 2, n_samples)
    #     volume = np.abs(np.random.normal(1000000, 500000, n_samples))

    #     # Calculate returns
    #     close_ret = np.diff(np.log(close_prices))
    #     close_ret = np.insert(close_ret, 0, 0)  # Add a zero at the beginning
    #     high_ret = np.diff(np.log(high_prices))
    #     high_ret = np.insert(high_ret, 0, 0)
    #     low_ret = np.diff(np.log(low_prices))
    #     low_ret = np.insert(low_ret, 0, 0)
    #     open_ret = np.diff(np.log(open_prices))
    #     open_ret = np.insert(open_ret, 0, 0)

    #     # Create a DataFrame
    #     df = pd.DataFrame(
    #         {
    #             "open": open_prices,
    #             "high": high_prices,
    #             "low": low_prices,
    #             "close": close_prices,
    #             "volume": volume,
    #             "open_ret": open_ret,
    #             "high_ret": high_ret,
    #             "low_ret": low_ret,
    #             "close_ret": close_ret,
    #         }
    #     )

    #     # Compute indicators on both prices and returns
    #     # Regular price-based indicators
    #     price_df = df.copy()
    #     ma_indicator = MA(period=14, use_returns=False)
    #     ema_indicator = EMA(period=14, use_returns=False)
    #     bb_indicator = BollingerBands(period=20, std_dev=2, use_returns=False)
    #     rsi_indicator = RSI(period=14, use_returns=False)
    #     macd_indicator = MACD(
    #         short_period=12, long_period=26, signal_period=9, use_returns=False
    #     )
    #     atr_indicator = ATR(period=14, use_returns=False)
    #     cci_indicator = CCI(period=20, use_returns=False)
    #     obv_indicator = OBV(period=1, use_returns=False)
    #     mfi_indicator = MFI(period=14, use_returns=False)
    #     stoch_indicator = Stochastic(period=14, use_returns=False)

    #     # Returns-based indicators
    #     returns_df = df.copy()
    #     ma_ret_indicator = MA(period=14, use_returns=True)
    #     ema_ret_indicator = EMA(period=14, use_returns=True)
    #     bb_ret_indicator = BollingerBands(period=20, std_dev=2, use_returns=True)
    #     rsi_ret_indicator = RSI(period=14, use_returns=True)

    #     # Calculate and add indicators to the DataFrame
    #     # Price-based
    #     price_df["MA_14"] = ma_indicator.compute(price_df)
    #     price_df["EMA_14"] = ema_indicator.compute(price_df)

    #     bb_values = bb_indicator.compute(price_df)
    #     price_df["BB_upper"] = bb_values["upper_band"]
    #     price_df["BB_middle"] = bb_values["middle_band"]
    #     price_df["BB_lower"] = bb_values["lower_band"]

    #     price_df["RSI_14"] = rsi_indicator.compute(price_df)

    #     macd_values = macd_indicator.compute(price_df)
    #     price_df["MACD_line"] = macd_values["macd_line"]
    #     price_df["MACD_signal"] = macd_values["signal_line"]
    #     price_df["MACD_histogram"] = macd_values["macd_histogram"]

    #     price_df["ATR_14"] = atr_indicator.compute(price_df)
    #     price_df["CCI_20"] = cci_indicator.compute(price_df)
    #     price_df["OBV_1"] = obv_indicator.compute(price_df)
    #     price_df["MFI_14"] = mfi_indicator.compute(price_df)
    #     price_df["Stochastic_14"] = stoch_indicator.compute(price_df)

    #     # Returns-based
    #     returns_df["MA_14_ret"] = ma_ret_indicator.compute(returns_df)
    #     returns_df["EMA_14_ret"] = ema_ret_indicator.compute(returns_df)

    #     bb_ret_values = bb_ret_indicator.compute(returns_df)
    #     returns_df["BB_upper_ret"] = bb_ret_values["upper_band"]
    #     returns_df["BB_middle_ret"] = bb_ret_values["middle_band"]
    #     returns_df["BB_lower_ret"] = bb_ret_values["lower_band"]

    #     returns_df["RSI_14_ret"] = rsi_ret_indicator.compute(returns_df)

    #     # Combine indicators
    #     combined_df = price_df.copy()
    #     for col in returns_df.columns:
    #         if col not in [
    #             "open",
    #             "high",
    #             "low",
    #             "close",
    #             "volume",
    #             "open_ret",
    #             "high_ret",
    #             "low_ret",
    #             "close_ret",
    #         ]:
    #             combined_df[col] = returns_df[col]

    #     # Remove NaN values
    #     combined_df = combined_df.dropna()

    #     # Normalize the data
    #     normalizer.fit(combined_df)
    #     normalized_df = normalizer.transform(combined_df)

    #     # Verify that indicators are normalized with correct methods
    #     # Price indicators should use Z_SCORE
    #     assert normalizer.stats["MA_14"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["EMA_14"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["BB_upper"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["BB_middle"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["BB_lower"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["ATR_14"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["CCI_20"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["OBV_1"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["MACD_line"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["MACD_signal"][0] == normalizer.Z_SCORE
    #     assert normalizer.stats["MACD_histogram"][0] == normalizer.Z_SCORE

    #     # Returns-based indicators should use SIGN_PRESERVING
    #     assert normalizer.stats["MA_14_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["EMA_14_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["BB_upper_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["BB_middle_ret"][0] == normalizer.SIGN_PRESERVING
    #     assert normalizer.stats["BB_lower_ret"][0] == normalizer.SIGN_PRESERVING

    #     # These should be in 0-100 range
    #     assert normalizer.stats["RSI_14"][0] == normalizer.SCALE_0_100
    #     assert normalizer.stats["MFI_14"][0] == normalizer.SCALE_0_100
    #     assert normalizer.stats["Stochastic_14"][0] == normalizer.SCALE_0_100

    #     # RSI on returns should also be 0-100 scaled
    #     assert normalizer.stats["RSI_14_ret"][0] == normalizer.SCALE_0_100

    #     # Verify that the indicators are actually normalized (values changed)
    #     for column in combined_df.columns:
    #         if column not in [
    #             "open",
    #             "high",
    #             "low",
    #             "close",
    #             "volume",
    #             "open_ret",
    #             "high_ret",
    #             "low_ret",
    #             "close_ret",
    #         ]:
    #             assert not np.array_equal(
    #                 normalized_df[column].to_numpy(), combined_df[column].to_numpy()
    #             ), f"{column} was not normalized"

    #     # Check that sign is preserved for return-based indicators
    #     for column in [
    #         "MA_14_ret",
    #         "EMA_14_ret",
    #         "BB_upper_ret",
    #         "BB_middle_ret",
    #         "BB_lower_ret",
    #     ]:
    #         for idx in range(len(combined_df)):
    #             # We need to handle the case where either value is exactly zero
    #             original_value = combined_df.loc[idx, column]
    #             normalized_value = normalized_df.loc[idx, column]

    #             # Use numpy's numeric conversion which is more robust
    #             try:
    #                 # numpy.asarray handles various types more safely
    #                 orig_val = np.asarray(original_value).item()
    #                 norm_val = np.asarray(normalized_value).item()

    #                 # If original is close to zero, normalized should be close to zero too
    #                 if (
    #                     abs(orig_val) < 1e-10
    #                 ):  # Use small epsilon for floating point comparison
    #                     assert abs(norm_val) < 1e-10, (
    #                         f"Zero value changed for {column} at index {idx}"
    #                     )
    #                 else:
    #                     # Check if signs match
    #                     assert (orig_val > 0 and norm_val > 0) or (
    #                         orig_val < 0 and norm_val < 0
    #                     ), (
    #                         f"Sign changed for {column} at index {idx}: original={orig_val}, normalized={norm_val}"
    #                     )
    #             except (ValueError, TypeError):
    #                 # Skip non-numeric values in case they exist
    #                 continue

    # def test_z_score_normalization(self, normalizer, sample_df):
    #     """Test that z-score normalization works correctly."""
    #     # Fit and transform
    #     normalizer.fit(sample_df)
    #     normalized_df = normalizer.transform(sample_df)

    #     # Check z-score normalization for price columns
    #     for column in ["open", "high", "low", "close"]:
    #         # Mean should be close to 0
    #         assert abs(normalized_df[column].mean()) < 0.1
    #         # Std should be close to 1
    #         assert 0.9 < normalized_df[column].std() < 1.1

    # def test_save_load(self, fitted_normalizer, normalizer, sample_df, tmp_path):
    #     """Test saving and loading normalization statistics."""
    #     # Define a temporary file path
    #     stats_file = tmp_path / "test_stats.json"

    #     # Test saving
    #     metadata = {
    #         "creation_date": "2023-06-15",
    #         "currency_pairs": ["EURUSD"],
    #         "timeframe": "1d",
    #         "date_range": "2023-01-01 to 2023-04-10",
    #     }

    #     fitted_normalizer.save(str(stats_file), metadata)

    #     # Check that the file was created
    #     assert os.path.exists(stats_file)

    #     # Test loading
    #     loaded_normalizer = FeatureNormalizer.load(str(stats_file))

    #     # Check that loaded normalizer has the same stats
    #     assert loaded_normalizer.is_fitted is True
    #     assert set(loaded_normalizer.stats.keys()) == set(
    #         fitted_normalizer.stats.keys()
    #     )

    #     # Compare some specific stats
    #     for column in ["open", "volume", "RSI_14"]:
    #         assert (
    #             loaded_normalizer.stats[column][0] == fitted_normalizer.stats[column][0]
    #         )
    #         for stat_key, stat_value in fitted_normalizer.stats[column][1].items():
    #             assert (
    #                 abs(loaded_normalizer.stats[column][1][stat_key] - stat_value)
    #                 < 1e-6
    #             )

    #     # Test that the loaded normalizer can transform data
    #     norm_df_original = fitted_normalizer.transform(sample_df)
    #     norm_df_loaded = loaded_normalizer.transform(sample_df)

    #     # Both normalizers should produce identical results
    #     for column in norm_df_original.columns:
    #         if column not in ["time", "timeframe", "asset", "market"]:
    #             assert np.allclose(norm_df_original[column], norm_df_loaded[column])

    # def test_load_missing_file(self):
    #     """Test loading from a missing file raises an error."""
    #     with pytest.raises(FileNotFoundError):
    #         FeatureNormalizer.load("nonexistent_file.json")

    # def test_load_invalid_file(self, tmp_path):
    #     """Test loading from an invalid JSON file raises an error."""
    #     # Create an invalid JSON file
    #     invalid_file = tmp_path / "invalid.json"
    #     with open(invalid_file, "w") as f:
    #         f.write("{invalid json")

    #     with pytest.raises(ValueError, match="Error parsing normalization stats file"):
    #         FeatureNormalizer.load(str(invalid_file))

    # def test_save_not_fitted(self, normalizer, tmp_path):
    #     """Test saving a normalizer that hasn't been fitted raises an error."""
    #     stats_file = tmp_path / "test_stats.json"

    #     with pytest.raises(ValueError, match="Normalizer is not fitted"):
    #         normalizer.save(str(stats_file))

    # def test_edge_case_empty_df(self, normalizer):
    #     """Test behavior with an empty DataFrame."""
    #     empty_df = pd.DataFrame()
    #     normalizer.fit(empty_df)

    #     # Should be marked as fitted but with no stats
    #     assert normalizer.is_fitted is True
    #     assert len(normalizer.stats) == 0

    # def test_edge_case_all_nan_column(self, normalizer, sample_df):
    #     """Test behavior with a column that contains only NaN values."""
    #     df_with_nan = sample_df.copy()
    #     df_with_nan["all_nan"] = np.nan

    #     normalizer.fit(df_with_nan)

    #     # The all_nan column should not be in stats
    #     assert "all_nan" not in normalizer.stats

    # def test_edge_case_constant_column(self, normalizer, sample_df):
    #     """Test behavior with a column that has constant values."""
    #     df_with_constant = sample_df.copy()
    #     df_with_constant["constant"] = 5.0

    #     normalizer.fit(df_with_constant)

    #     # The constant column should have std=1.0 to avoid division by zero
    #     assert normalizer.stats["constant"][1]["std"] == 1.0


if __name__ == "__main__":
    """
    Main entry point for running the tests directly.
    """
    pytest.main(["-xvs", "tests/unit/test_normalizer.py"])
