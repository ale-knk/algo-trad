"""
Unit tests for the Window class of the PyTrad framework.

This test suite provides comprehensive coverage of the Window class,
which represents a collection of candlesticks for financial analysis.

The tests verify:
1. Correct initialization and attribute access
2. Factory methods for creating Windows from different data sources
3. Container-like behavior (__len__, __getitem__, __setitem__)
4. Data conversion methods (to_dict, to_df, to_tensors)
5. Indicator integration
6. Validation logic

Note: Tests for plotting functionality are deliberately excluded.

To run all tests:
    pytest -xvs tests/unit/test_window.py

To run a specific test class:
    pytest -xvs tests/unit/test_window.py::TestWindow

To run a specific test:
    pytest -xvs tests/unit/test_window.py::TestWindow::test_initialization
"""

import unittest.mock as mock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from pytrad.candle import Candle
from pytrad.indicator import MA, MACD
from pytrad.window import Window


@pytest.fixture
def mock_ma_data():
    """Mock data for MA indicator."""
    return [105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0, 121.0, 123.0]


@pytest.fixture
def mock_macd_data():
    """Mock data for MACD indicator."""
    return {
        "macd_line": [
            105.0,
            107.0,
            109.0,
            111.0,
            113.0,
            115.0,
            117.0,
            119.0,
            121.0,
            123.0,
        ],
        "signal_line": [
            105.0,
            107.0,
            109.0,
            111.0,
            113.0,
            115.0,
            117.0,
            119.0,
            121.0,
            123.0,
        ],
        "macd_histogram": [
            105.0,
            107.0,
            109.0,
            111.0,
            113.0,
            115.0,
            117.0,
            119.0,
            121.0,
            123.0,
        ],
    }


class TestWindow:
    """Test suite for the Window class."""

    @pytest.fixture
    def sample_candles(self):
        """Create a list of sample candles for testing."""
        candles = []
        base_time = datetime(2023, 1, 1, 12, 0, 0)

        for i in range(10):
            candle_time = base_time + timedelta(hours=i)
            candles.append(
                Candle(
                    time=candle_time,
                    open_price=100.0 + i,
                    high=105.0 + i,
                    low=95.0 + i,
                    close=102.0 + i,
                    volume=1000.0 + i * 100,
                    timeframe="1h",
                    asset="EURUSD",
                    market="forex",
                )
            )
        return candles

    @pytest.fixture
    def sample_window(self, sample_candles):
        """Create a sample window for testing."""
        return Window(candles=sample_candles)

    @pytest.fixture
    def sample_indicators(self, mock_ma_data, mock_macd_data):
        """Create a dictionary of sample indicators."""
        return {
            "MA_3": mock_ma_data,
            "MACD_12_26_9": mock_macd_data,
        }

    @pytest.fixture
    def sample_window_with_indicators(self, sample_candles, sample_indicators):
        """Create a sample window with indicators using the Window constructor."""
        return Window(candles=sample_candles, indicators=sample_indicators)

    @pytest.fixture
    def sample_window_with_nans(self, sample_candles, sample_indicators):
        """Create a sample window with NaNs in indicators."""
        # Add NaNs to the indicators
        sample_indicators["MA_3"][0] = np.nan
        sample_indicators["MACD_12_26_9"]["macd_line"][0] = np.nan
        sample_indicators["MACD_12_26_9"]["signal_line"][0] = np.nan
        sample_indicators["MACD_12_26_9"]["macd_histogram"][0] = np.nan
        return Window(candles=sample_candles, indicators=sample_indicators)

    @pytest.fixture
    def sample_df(self, sample_candles):
        """Create a sample DataFrame for testing from_df method."""
        data = []
        for candle in sample_candles:
            data.append(
                {
                    "time": candle.time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "timeframe": candle.timeframe,
                    "asset": candle.asset,
                    "market": candle.market,
                }
            )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_df_with_indicators(self, sample_df, mock_ma_data, mock_macd_data):
        """Create a sample DataFrame with mocked indicators."""
        # Add MA indicator
        sample_df["MA_5"] = mock_ma_data[
            : len(sample_df)
        ]  # Use the first N values of mock_ma_data

        # Add MACD indicator components
        sample_df["MACD_12_26_9-macd_line"] = mock_macd_data["macd_line"][
            : len(sample_df)
        ]
        sample_df["MACD_12_26_9-signal_line"] = mock_macd_data["signal_line"][
            : len(sample_df)
        ]
        sample_df["MACD_12_26_9-macd_histogram"] = mock_macd_data["macd_histogram"][
            : len(sample_df)
        ]

        # Add RSI indicator (if needed, you can mock this as well)
        sample_df["RSI_14-returns"] = np.random.randn(
            len(sample_df)
        )  # Keep this random for now

        return sample_df

    @pytest.fixture
    def sample_dict(self, sample_df):
        """Create a sample dict with Series objects for testing from_dict method."""
        return {i: row for i, row in sample_df.iterrows()}

    def test_initialization(self, sample_candles):
        """Test that Window initializes correctly with a list of candles."""
        window = Window(candles=sample_candles)

        # Check basic properties
        assert len(window.candles) == len(sample_candles)
        assert window.asset == "EURUSD"
        assert window.market == "forex"
        assert window.timeframe == "1h"

        # Check that the candles are stored correctly
        for i, candle in enumerate(sample_candles):
            assert window.candles[i] is candle

        # Check that indicators are initialized as empty
        assert window.indicators == {}

    def test_validate_successful(self, sample_candles):
        """Test that validate method passes for valid candles."""
        window = Window(candles=sample_candles)
        # If we get here, validation succeeded
        assert True

    def test_validate_missing_market_asset(self, sample_candles):
        """Test that validate raises error when market or asset is missing."""
        # Create a candle with missing market
        bad_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
            asset="EURUSD",
            market=None,  # Missing market
        )

        bad_candles = sample_candles.copy()
        bad_candles[3] = bad_candle

        with pytest.raises(ValueError, match="must have a market and asset"):
            Window(candles=bad_candles)

    def test_validate_different_market(self, sample_candles):
        """Test that validate raises error when candles have different markets."""
        # Create a candle with different market
        different_market_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
            asset="EURUSD",
            market="crypto",  # Different market
        )

        mixed_candles = sample_candles.copy()
        mixed_candles[3] = different_market_candle

        with pytest.raises(ValueError, match="same market"):
            Window(candles=mixed_candles)

    def test_validate_different_asset(self, sample_candles):
        """Test that validate raises error when candles have different assets."""
        # Create a candle with different asset
        different_asset_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
            asset="GBPUSD",  # Different asset
            market="forex",
        )

        mixed_candles = sample_candles.copy()
        mixed_candles[3] = different_asset_candle

        with pytest.raises(ValueError, match="same asset"):
            Window(candles=mixed_candles)

    def test_validate_different_timeframe(self, sample_candles):
        """Test that validate raises error when candles have different timeframes."""
        # Create a candle with different timeframe
        different_tf_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="4h",  # Different timeframe
            asset="EURUSD",
            market="forex",
        )

        mixed_candles = sample_candles.copy()
        mixed_candles[3] = different_tf_candle

        with pytest.raises(ValueError, match="same timeframe"):
            Window(candles=mixed_candles)

    def test_len(self, sample_candles, sample_window):
        """Test __len__ method returns correct number of candles."""
        assert len(sample_window) == len(sample_candles)

    def test_getitem_index(self, sample_window, sample_candles):
        """Test __getitem__ method with integer index."""
        for i in range(len(sample_candles)):
            assert sample_window[i] is sample_candles[i]

        # Test negative indexing
        assert sample_window[-1] is sample_candles[-1]

    def test_getitem_slice(self, sample_window, sample_candles):
        """Test __getitem__ method with slice."""
        sliced_window = sample_window[2:5]

        # Check it returns a Window instance
        assert isinstance(sliced_window, Window)

        # Check it has the correct candles
        assert len(sliced_window) == 3
        for i in range(3):
            assert sliced_window.candles[i] is sample_candles[2 + i]

    def test_getitem_slice_preserve_indicators(self, sample_window_with_indicators):
        """Test __getitem__ method with slice preserves indicators."""
        sliced_window = sample_window_with_indicators[2:5]

        # Check it returns a Window instance
        assert isinstance(sliced_window, Window)

        # Check it has the correct candles
        assert len(sliced_window) == 3

        # Check indicators are sliced correctly
        assert len(sliced_window.indicators["MA_3"]) == 3
        assert len(sliced_window.indicators["MACD_12_26_9"]["macd_line"]) == 3
        assert len(sliced_window.indicators["MACD_12_26_9"]["signal_line"]) == 3
        assert len(sliced_window.indicators["MACD_12_26_9"]["macd_histogram"]) == 3

    def test_getitem_invalid(self, sample_window):
        """Test __getitem__ method with invalid type."""
        with pytest.raises(TypeError, match="Invalid argument type"):
            _ = sample_window["invalid"]

    def test_setitem_valid(self, sample_window, sample_candles):
        """Test __setitem__ method with valid candle."""
        new_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=110.0,
            high=115.0,
            low=105.0,
            close=112.0,
            volume=2000.0,
            timeframe="1h",
            asset="EURUSD",
            market="forex",
        )

        sample_window[3] = new_candle
        assert sample_window[3] is new_candle

    def test_setitem_negative_index(self, sample_window, sample_candles):
        """Test __setitem__ method with negative index."""
        new_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=110.0,
            high=115.0,
            low=105.0,
            close=112.0,
            volume=2000.0,
            timeframe="1h",
            asset="EURUSD",
            market="forex",
        )

        sample_window[-1] = new_candle
        assert sample_window[9] is new_candle
        assert sample_window[-1] is new_candle

    def test_setitem_out_of_range(self, sample_window):
        """Test __setitem__ method with index out of range."""
        new_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=110.0,
            high=115.0,
            low=105.0,
            close=112.0,
            volume=2000.0,
            timeframe="1h",
            asset="EURUSD",
            market="forex",
        )

        with pytest.raises(IndexError, match="Index out of range"):
            sample_window[20] = new_candle

        with pytest.raises(IndexError, match="Index out of range"):
            sample_window[-20] = new_candle

    def test_setitem_invalid_type(self, sample_window):
        """Test __setitem__ method with invalid value type."""
        with pytest.raises(TypeError, match="Value must be a Candle instance"):
            sample_window[3] = "not a candle"

    def test_setitem_invalid_index_type(self, sample_window):
        """Test __setitem__ method with invalid index type."""
        new_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=110.0,
            high=115.0,
            low=105.0,
            close=112.0,
            volume=2000.0,
            timeframe="1h",
            asset="EURUSD",
            market="forex",
        )

        with pytest.raises(TypeError, match="Invalid argument type"):
            sample_window["invalid"] = new_candle

    def test_from_df_no_indicators(self, sample_df):
        """Test from_df class method."""
        window = Window.from_df(sample_df)

        # Check we got the right number of candles
        assert len(window) == len(sample_df)

        # Check properties of first candle
        first_row = sample_df.iloc[0]
        first_candle = window[0]

        assert first_candle.time == first_row["time"]
        assert first_candle.open == first_row["open"]
        assert first_candle.high == first_row["high"]
        assert first_candle.low == first_row["low"]
        assert first_candle.close == first_row["close"]
        assert first_candle.volume == first_row["volume"]
        assert first_candle.timeframe == first_row["timeframe"]
        assert first_candle.asset == first_row["asset"]
        assert first_candle.market == first_row["market"]

    def test_from_df_with_indicators(self, sample_df_with_indicators):
        """Test from_df class method with indicators."""
        window = Window.from_df(sample_df_with_indicators)

        # Basic length check
        assert len(window) == len(sample_df_with_indicators)

        # Check MA indicator
        assert "MA_5" in window.indicators
        assert len(window.indicators["MA_5"]) == len(window)

        # Check MACD indicator exists
        assert "MACD_12_26_9" in window.indicators

        # Check MACD components
        macd_components = window.indicators["MACD_12_26_9"]
        assert "macd_line" in macd_components
        assert "signal_line" in macd_components
        assert "macd_histogram" in macd_components

        # Check MACD component lengths
        assert len(macd_components["macd_line"]) == len(window)
        assert len(macd_components["signal_line"]) == len(window)
        assert len(macd_components["macd_histogram"]) == len(window)

        # Check MACD line values match DataFrame
        assert (
            macd_components["macd_line"]
            == sample_df_with_indicators["MACD_12_26_9-macd_line"]
        ).all()

        # Check signal line values match DataFrame
        assert (
            macd_components["signal_line"]
            == sample_df_with_indicators["MACD_12_26_9-signal_line"]
        ).all()

        # Check histogram values match DataFrame
        assert (
            macd_components["macd_histogram"]
            == sample_df_with_indicators["MACD_12_26_9-macd_histogram"]
        ).all()

    def test_add_indicator_simple(self, sample_window, mock_ma_data):
        """Test add_indicator method."""
        # Create a simple MA indicator
        ma = MA(period=3)
        with mock.patch.object(ma, "compute", return_value=mock_ma_data):
            sample_window.add_indicator(ma)

        # Check indicator was added
        assert "MA_3" in sample_window.indicators
        assert len(sample_window.indicators["MA_3"]) == len(mock_ma_data)

    def test_add_indicator_complex(self, sample_window, mock_macd_data):
        """Test add_indicator method."""
        # Create a MACD indicator
        macd = MACD()
        with mock.patch.object(macd, "compute", return_value=mock_macd_data):
            sample_window.add_indicator(macd)

        # Check indicator was added
        assert "MACD_12_26_9" in sample_window.indicators
        assert isinstance(sample_window.indicators["MACD_12_26_9"], dict)
        assert len(sample_window.indicators["MACD_12_26_9"]) == 3
        assert len(sample_window.indicators["MACD_12_26_9"]["macd_line"]) == len(
            mock_macd_data["macd_line"]
        )
        assert len(sample_window.indicators["MACD_12_26_9"]["signal_line"]) == len(
            mock_macd_data["signal_line"]
        )
        assert len(sample_window.indicators["MACD_12_26_9"]["macd_histogram"]) == len(
            mock_macd_data["macd_histogram"]
        )

    def test_normalize_basic(self, sample_window):
        """Test basic normalization with default normalizer."""
        from pytrad.normalizer import FeatureNormalizer

        # Create a normalizer
        normalizer = FeatureNormalizer()

        # Store original values for comparison
        original_values = {
            "open": [candle.open for candle in sample_window],
            "high": [candle.high for candle in sample_window],
            "low": [candle.low for candle in sample_window],
            "close": [candle.close for candle in sample_window],
            "volume": [candle.volume for candle in sample_window],
        }

        # Normalize the window
        normalized_window = sample_window.normalize(normalizer, inplace=False)

        assert normalizer.is_fitted
        assert len(sample_window) == len(normalized_window)

        # Check that values have changed
        for i, candle in enumerate(normalized_window):
            assert candle.open != original_values["open"][i]
            assert candle.high != original_values["high"][i]
            assert candle.low != original_values["low"][i]
            assert candle.close != original_values["close"][i]
            assert candle.volume != original_values["volume"][i]

        # Verify that metadata is preserved
        assert normalized_window.asset == sample_window.asset
        assert normalized_window.market == sample_window.market
        assert normalized_window.timeframe == sample_window.timeframe

    def test_normalize_with_nans(self, sample_window_with_nans):
        """Test normalization with NaN values in indicators."""
        from pytrad.normalizer import FeatureNormalizer

        normalizer = FeatureNormalizer()

        # Normalize window with NaNs
        normalized_window = sample_window_with_nans.normalize(normalizer, inplace=False)

        # Verify that NaNs were handled properly
        assert not np.isnan(normalized_window.indicators["MA_3"]).any()
        assert not np.isnan(
            normalized_window.indicators["MACD_12_26_9"]["macd_line"]
        ).any()

    def test_normalize_inplace(self, sample_window):
        """Test inplace normalization."""
        from pytrad.normalizer import FeatureNormalizer

        normalizer = FeatureNormalizer()

        # Store original values
        original_values = [candle.open for candle in sample_window]

        # Normalize inplace
        result = sample_window.normalize(normalizer, inplace=True)

        # Verify that values changed and same instance returned
        assert result is sample_window
        assert [candle.open for candle in sample_window] != original_values

    def test_normalize_preserves_indicators(self, sample_window_with_indicators):
        """Test that normalization preserves indicator structure."""
        from pytrad.normalizer import FeatureNormalizer

        normalizer = FeatureNormalizer()

        # Normalize window with indicators
        normalized_window = sample_window_with_indicators.normalize(
            normalizer, inplace=False
        )

        # Verify indicator structure is preserved
        assert set(normalized_window.indicators.keys()) == set(
            sample_window_with_indicators.indicators.keys()
        )
        for indicator_name, values in normalized_window.indicators.items():
            if isinstance(values, dict):
                assert set(values.keys()) == set(
                    sample_window_with_indicators.indicators[indicator_name].keys()
                )

    def test_normalize_preserves_indicators_inplace(
        self, sample_window, sample_indicators
    ):
        """Test that inplace normalization preserves indicator structure and values."""
        from pytrad.normalizer import FeatureNormalizer

        # Add indicators to the window
        sample_window.indicators = sample_indicators

        # Store original indicator values
        original_indicators = {
            "MA_3": sample_indicators["MA_3"].copy(),
            "MACD_12_26_9": {
                "macd_line": sample_indicators["MACD_12_26_9"]["macd_line"].copy(),
                "signal_line": sample_indicators["MACD_12_26_9"]["signal_line"].copy(),
                "macd_histogram": sample_indicators["MACD_12_26_9"][
                    "macd_histogram"
                ].copy(),
            },
        }

        # Create and apply normalizer
        normalizer = FeatureNormalizer()
        result = sample_window.normalize(normalizer, inplace=True)

        # Verify that the same instance is returned
        assert result is sample_window

        # Verify indicator structure is preserved
        assert set(sample_window.indicators.keys()) == set(original_indicators.keys())
        for indicator_name, values in sample_window.indicators.items():
            if isinstance(values, dict):
                assert set(values.keys()) == set(
                    original_indicators[indicator_name].keys()
                )

        # Verify that indicator values have changed (normalization should modify values)
        assert sample_window.indicators["MA_3"] != original_indicators["MA_3"]
        assert (
            sample_window.indicators["MACD_12_26_9"]["macd_line"]
            != original_indicators["MACD_12_26_9"]["macd_line"]
        )
        assert (
            sample_window.indicators["MACD_12_26_9"]["signal_line"]
            != original_indicators["MACD_12_26_9"]["signal_line"]
        )
        assert (
            sample_window.indicators["MACD_12_26_9"]["macd_histogram"]
            != original_indicators["MACD_12_26_9"]["macd_histogram"]
        )

    # def test_normalize_multiple_calls(self, sample_window):
    #     """Test multiple normalization calls."""
    #     from pytrad.normalizer import FeatureNormalizer

    #     # Create and fit normalizer with original data
    #     normalizer = FeatureNormalizer()
    #     original_df = sample_window.to_df()
    #     normalizer.fit(original_df)

    #     # First normalization
    #     first_normalized = sample_window.normalize(normalizer, inplace=False)

    #     # Verify first normalization
    #     first_df = first_normalized.to_df()
    #     assert not first_df.isnull().values.any()  # No NaN values
    #     assert not np.isinf(first_df.values).any()  # No infinite values

    #     # Second normalization using the same fitted normalizer
    #     second_normalized = first_normalized.normalize(normalizer, inplace=False)

    #     # Verify second normalization
    #     second_df = second_normalized.to_df()
    #     assert not second_df.isnull().values.any()
    #     assert not np.isinf(second_df.values).any()

    #     # Verify that values are stable after second normalization
    #     for i in range(len(first_normalized)):
    #         assert np.isclose(
    #             first_normalized[i].open, second_normalized[i].open, rtol=1e-5
    #         )
    #         assert np.isclose(
    #             first_normalized[i].high, second_normalized[i].high, rtol=1e-5
    #         )
    #         assert np.isclose(
    #             first_normalized[i].low, second_normalized[i].low, rtol=1e-5
    #         )
    #         assert np.isclose(
    #             first_normalized[i].close, second_normalized[i].close, rtol=1e-5
    #         )
    #         assert np.isclose(
    #             first_normalized[i].volume, second_normalized[i].volume, rtol=1e-5
    #         )

    def test_clear_indicators_nans_not_inplace(self, sample_window):
        """Test clearing NaN values without inplace modification."""
        # Add some NaN values at the beginning of indicators
        sample_window.indicators["MA_3"] = [
            np.nan,
            np.nan,
            105.0,
            107.0,
            109.0,
            111.0,
            113.0,
            115.0,
            117.0,
            119.0,
        ]
        sample_window.indicators["MACD_12_26_9"] = {
            "macd_line": [
                np.nan,
                np.nan,
                105.0,
                107.0,
                109.0,
                111.0,
                113.0,
                115.0,
                117.0,
                119.0,
            ],
            "signal_line": [
                np.nan,
                np.nan,
                105.0,
                107.0,
                109.0,
                111.0,
                113.0,
                115.0,
                117.0,
                119.0,
            ],
            "macd_histogram": [
                np.nan,
                np.nan,
                105.0,
                107.0,
                109.0,
                111.0,
                113.0,
                115.0,
                117.0,
                119.0,
            ],
        }

        # Store original length
        original_length = len(sample_window)

        # Clear NaNs without inplace
        cleaned_window = sample_window.clear_indicators_nans(inplace=False)
        print("CLEANED WINDOW LEN", len(cleaned_window))
        print(len(cleaned_window.indicators["MA_3"]))
        # Check that a new instance is returned
        assert cleaned_window is not sample_window

        # Check that NaNs were removed in the new window
        assert len(cleaned_window.indicators["MA_3"]) == original_length - 2
        assert (
            len(cleaned_window.indicators["MACD_12_26_9"]["macd_line"])
            == original_length - 2
        )
        assert cleaned_window.indicators["MA_3"][0] == 105.0
        assert cleaned_window.indicators["MACD_12_26_9"]["macd_line"][0] == 105.0

        # Check that original window remains unchanged
        assert len(sample_window.indicators["MA_3"]) == original_length
        assert (
            len(sample_window.indicators["MACD_12_26_9"]["macd_line"])
            == original_length
        )

    def test_clear_indicators_nans_no_nans(self, sample_window_with_indicators):
        """Test clearing NaN values when there are none."""
        # Store original indicators
        original_indicators = {
            "MA_3": sample_window_with_indicators.indicators["MA_3"].copy(),
            "MACD_12_26_9": {
                "macd_line": sample_window_with_indicators.indicators["MACD_12_26_9"][
                    "macd_line"
                ].copy(),
                "signal_line": sample_window_with_indicators.indicators["MACD_12_26_9"][
                    "signal_line"
                ].copy(),
                "macd_histogram": sample_window_with_indicators.indicators[
                    "MACD_12_26_9"
                ]["macd_histogram"].copy(),
            },
        }

        # Clear NaNs
        result = sample_window_with_indicators.clear_indicators_nans(inplace=True)

        # Check that the same instance is returned
        assert result is sample_window_with_indicators

        # Check that indicators remain unchanged
        assert (
            sample_window_with_indicators.indicators["MA_3"]
            == original_indicators["MA_3"]
        )
        assert (
            sample_window_with_indicators.indicators["MACD_12_26_9"]["macd_line"]
            == original_indicators["MACD_12_26_9"]["macd_line"]
        )

    # def test_clear_indicators_nans_empty_window(self):
    #     """Test clearing NaN values in an empty window."""
    #     # Create empty window
    #     window = Window(candles=[])

    #     # Clear NaNs
    #     result = window.clear_indicators_nans(inplace=True)

    #     # Check that the same instance is returned
    #     assert result is window

    #     # Check that window remains empty
    #     assert len(result) == 0

    def test_to_tensor(self, sample_window_with_indicators):
        """Test the to_tensors method of the Window class with indicators."""
        # Convert window to tensors
        tensors = sample_window_with_indicators.to_tensors()

        # Verify basic tensor structure
        assert isinstance(tensors, dict)
        assert set(tensors.keys()) == {
            "ohlc",
            "ohlc_ret",
            "indicators",
            "time",
            "market",
            "asset",
            "timeframe",
        }

        # Verify OHLC tensor
        assert isinstance(tensors["ohlc"], torch.Tensor)
        assert tensors["ohlc"].shape == (
            len(sample_window_with_indicators),
            4,
        )  # [sequence_length, 4 features]

        # Verify OHLC returns tensor
        assert isinstance(tensors["ohlc_ret"], torch.Tensor)
        assert tensors["ohlc_ret"].shape == (
            len(sample_window_with_indicators),
            4,
        )  # [sequence_length, 4 features]

        # Verify indicators tensor
        assert isinstance(tensors["indicators"], torch.Tensor)
        # Shape should be [sequence_length, n_indicators]
        assert len(tensors["indicators"].shape) == 2
        assert tensors["indicators"].shape[0] == len(sample_window_with_indicators)
        assert tensors["indicators"].shape[1] > 0  # Should have at least one indicator

        # Verify time features tensor
        assert isinstance(tensors["time"], torch.Tensor)
        assert tensors["time"].shape == (
            len(sample_window_with_indicators),
            19,
        )  # 19 time features

        # Verify market, asset, and timeframe tensors
        for key in ["market", "asset", "timeframe"]:
            assert isinstance(tensors[key], torch.Tensor)
            assert tensors[key].shape == (1,)  # Single value for each

        # Verify values in OHLC tensor match candle data
        for i, candle in enumerate(sample_window_with_indicators):
            assert torch.allclose(
                tensors["ohlc"][i],
                torch.tensor(
                    [candle.open, candle.high, candle.low, candle.close],
                    dtype=torch.float32,
                ),
            )

        # Verify returns calculation
        for i in range(1, len(sample_window_with_indicators)):
            prev_close = sample_window_with_indicators[i - 1].close
            curr_close = sample_window_with_indicators[i].close
            expected_return = (
                np.log(curr_close / prev_close) if prev_close != 0 else 0.0
            )
            assert torch.allclose(
                tensors["ohlc_ret"][i, 3],  # Close return is at index 3
                torch.tensor(expected_return, dtype=torch.float32),
            )

        # Verify time features
        first_candle = sample_window_with_indicators[0]
        time_features = sample_window_with_indicators._extract_time_features(
            first_candle
        )
        assert torch.allclose(
            tensors["time"][0],
            torch.tensor(time_features, dtype=torch.float32),
        )

        # Verify indicator values
        # Check MA indicator
        ma_values = sample_window_with_indicators.indicators["MA_3"]
        ma_tensor = tensors["indicators"][:, 0]  # First indicator column
        assert torch.allclose(
            ma_tensor[
                -len(ma_values) :
            ],  # Only check the last N values (MA has fewer values)
            torch.tensor(ma_values, dtype=torch.float32),
        )

        # Check MACD components
        macd_values = sample_window_with_indicators.indicators["MACD_12_26_9"]
        macd_line_tensor = tensors["indicators"][:, 1]  # Second indicator column
        signal_line_tensor = tensors["indicators"][:, 2]  # Third indicator column
        histogram_tensor = tensors["indicators"][:, 3]  # Fourth indicator column

        assert torch.allclose(
            macd_line_tensor[-len(macd_values["macd_line"]) :],
            torch.tensor(macd_values["macd_line"], dtype=torch.float32),
        )
        assert torch.allclose(
            signal_line_tensor[-len(macd_values["signal_line"]) :],
            torch.tensor(macd_values["signal_line"], dtype=torch.float32),
        )
        assert torch.allclose(
            histogram_tensor[-len(macd_values["macd_histogram"]) :],
            torch.tensor(macd_values["macd_histogram"], dtype=torch.float32),
        )


if __name__ == "__main__":
    """
    Main entry point for running the tests directly.
    """
    pytest.main(["-xvs", "tests/unit/test_window.py"])
