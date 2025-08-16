"""
Unit tests for the dataset classes of the PyTrad framework.

This test suite provides comprehensive coverage of the WindowDataset class,
which manages time series data for machine learning.

The tests verify:
1. Correct initialization with different parameters
2. Data loading from the database
3. Indicator computation and configuration
4. NaN handling in indicators
5. Normalization of data
6. Retrieval of window slices by datetime

To run all tests:
    pytest -xvs tests/unit/test_dataset.py

To run a specific test class:
    pytest -xvs tests/unit/test_dataset.py::TestWindowDataset

To run a specific test:
    pytest -xvs tests/unit/test_dataset.py::TestWindowDataset::test_initialization
"""

import unittest.mock as mock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pytrad.candle import Candle
from pytrad.dataset import WindowDataset
from pytrad.indicator import MA, RSI
from pytrad.normalizer import FeatureNormalizer
from pytrad.window import Window


class TestWindowDataset:
    """Test suite for the WindowDataset class."""

    @pytest.fixture
    def sample_candles(self):
        """Create a list of sample candles for testing."""
        candles = []
        base_time = datetime(2023, 1, 1, 12, 0, 0)

        for i in range(30):
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
    def mock_from_db(self, sample_window):
        """Create a mock for Window.from_db that returns a sample window."""
        with mock.patch(
            "pytrad.window.Window.from_db", return_value=sample_window
        ) as mock_from_db:
            yield mock_from_db

    @pytest.fixture
    def mock_normalizer(self):
        """Create a mock for FeatureNormalizer."""
        normalizer = mock.MagicMock(spec=FeatureNormalizer)
        normalizer.is_fitted = False
        normalizer.fit.return_value = None
        normalizer.transform.return_value = pd.DataFrame(
            {
                "time": [c.time for c in range(30)],
                "open": [100.0 + i for i in range(30)],
                "high": [105.0 + i for i in range(30)],
                "low": [95.0 + i for i in range(30)],
                "close": [102.0 + i for i in range(30)],
                "volume": [1000.0 + i * 100 for i in range(30)],
                "timeframe": ["1h" for _ in range(30)],
                "asset": ["EURUSD" for _ in range(30)],
                "market": ["forex" for _ in range(30)],
            }
        )
        return normalizer

    @pytest.fixture
    def dataset_params(self):
        """Return basic parameters for creating a WindowDataset."""
        return {
            "asset": "EURUSD",
            "timeframe": "1h",
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 1, 2),
            "window_size": 10,
            "stride": 1,
            "normalize": False,  # Disable normalization by default for tests
        }

    def test_initialization(self, mock_from_db, dataset_params):
        """Test that WindowDataset initializes correctly with basic parameters."""
        dataset = WindowDataset(**dataset_params)

        # Check basic properties
        assert dataset.asset == "EURUSD"
        assert dataset.timeframe == "1h"
        assert dataset.start_date == datetime(2023, 1, 1)
        assert dataset.end_date == datetime(2023, 1, 2)
        assert dataset.window_size == 10
        assert dataset.stride == 1
        assert dataset.indicators == []
        assert dataset.normalize is False

        # Check that Window.from_db was called with correct params
        mock_from_db.assert_called_once_with(
            asset="EURUSD",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
        )

    def test_initialization_with_indicators(
        self, mock_from_db, dataset_params, sample_window
    ):
        """Test initialization with indicators."""
        # Create indicators
        ma = MA(period=5)
        rsi = RSI(period=14)

        # Add indicators to params
        params = dataset_params.copy()
        params["indicators"] = [ma, rsi]

        # Mock indicator computation to avoid actual calculations
        with mock.patch.object(Window, "add_indicators") as mock_add_indicators:
            dataset = WindowDataset(**params)

            # Check indicators were properly set
            assert len(dataset.indicators) == 2
            assert dataset.indicators[0] == ma
            assert dataset.indicators[1] == rsi

            # Check add_indicators was called
            mock_add_indicators.assert_called_once_with([ma, rsi])

    def test_load_window(self, mock_from_db, dataset_params):
        """Test _load_window method."""
        dataset = WindowDataset(**dataset_params)

        # Check Window.from_db was called with correct params
        mock_from_db.assert_called_once_with(
            asset="EURUSD",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
        )

        # Check window is properly stored
        assert dataset.window is not None
        assert dataset.window == mock_from_db.return_value

    def test_compute_indicators(self, mock_from_db, dataset_params, sample_window):
        """Test _compute_indicators method."""
        # Create indicators
        ma = MA(period=5)
        rsi = RSI(period=14)

        # Add indicators to params
        params = dataset_params.copy()
        params["indicators"] = [ma, rsi]

        # Mock methods to avoid actual calculations
        with mock.patch.object(Window, "add_indicators") as mock_add_indicators:
            with mock.patch.object(
                WindowDataset, "_clean_indicator_nans"
            ) as mock_clean:
                dataset = WindowDataset(**params)

                # Check methods were called
                mock_add_indicators.assert_called_once_with([ma, rsi])
                mock_clean.assert_called_once()

    def test_configure_indicators(self, mock_from_db, dataset_params):
        """Test _configure_indicators method."""
        # Create indicators that should be configured
        ma = MA(period=5)
        bb = mock.MagicMock()
        bb.name = "BB_20"
        bb.use_returns = False

        other_indicator = mock.MagicMock()
        other_indicator.name = "Other"
        other_indicator.use_returns = False

        # Add indicators to params
        params = dataset_params.copy()
        params["indicators"] = [ma, bb, other_indicator]

        # Mock to avoid actual indicator computation
        with mock.patch.object(Window, "add_indicators"):
            dataset = WindowDataset(**params)

            # Check use_returns has been set correctly
            assert ma.use_returns is True  # Should be set to True for MA
            assert bb.use_returns is True  # Should be set to True for BB
            assert (
                other_indicator.use_returns is False
            )  # Should remain False for other indicators

    def test_clean_indicator_nans(self, mock_from_db, dataset_params, sample_window):
        """Test _clean_indicator_nans method with NaN values."""
        # Create a DataFrame with some NaN values at the beginning
        df = pd.DataFrame(
            {
                "time": [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)],
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000.0 + i * 100 for i in range(10)],
                "indicator": [
                    np.nan,
                    np.nan,
                    np.nan,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                ],
            }
        )

        with mock.patch.object(Window, "to_df", return_value=df):
            with mock.patch.object(Window, "__getitem__") as mock_getitem:
                dataset = WindowDataset(**dataset_params)
                dataset._clean_indicator_nans()

                # Check that window slicing was called with first_valid_idx = 3
                # (the first index where all values are valid)
                mock_getitem.assert_called_once_with(slice(3, None))

    def test_setup_normalizer(self, mock_from_db, dataset_params):
        """Test _setup_normalizer method."""
        # Test with normalize=True but no provided normalizer
        params = dataset_params.copy()
        params["normalize"] = True

        dataset = WindowDataset(**params)

        # Normalizer should be created
        assert dataset.normalizer is not None
        assert isinstance(dataset.normalizer, FeatureNormalizer)
        assert dataset.normalizer.max_samples_for_stats == 10000  # Default value

        # Test with provided normalizer
        custom_normalizer = FeatureNormalizer(max_samples_for_stats=5000)
        params["normalizer"] = custom_normalizer

        dataset = WindowDataset(**params)

        # Should use the provided normalizer
        assert dataset.normalizer is custom_normalizer
        assert dataset.normalizer.max_samples_for_stats == 5000

    def test_fit_transform_normalizer(
        self, mock_from_db, dataset_params, sample_window
    ):
        """Test _fit_transform_normalizer method."""
        # Create params with normalize=True
        params = dataset_params.copy()
        params["normalize"] = True

        # Create a mock normalizer
        mock_normalizer = mock.MagicMock(spec=FeatureNormalizer)
        mock_normalizer.is_fitted = False
        mock_normalizer.transform.return_value = pd.DataFrame()
        params["normalizer"] = mock_normalizer

        # Mock Window.from_df to avoid actual creation
        with mock.patch.object(Window, "from_df") as mock_from_df:
            dataset = WindowDataset(**params)

            # Check normalizer methods were called
            mock_normalizer.fit.assert_called_once()
            mock_normalizer.transform.assert_called_once()
            mock_from_df.assert_called_once()

        # Test with already fitted normalizer
        mock_normalizer = mock.MagicMock(spec=FeatureNormalizer)
        mock_normalizer.is_fitted = True
        params["normalizer"] = mock_normalizer

        with mock.patch.object(Window, "from_df"):
            with pytest.warns(UserWarning, match="Normalizer is already fitted"):
                dataset = WindowDataset(**params)

                # fit should not be called when normalizer is already fitted
                mock_normalizer.fit.assert_not_called()

    def test_getitem(self, mock_from_db, dataset_params, sample_window):
        """Test __getitem__ method."""
        # Create a window with 30 candles at hourly intervals
        candles = []
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        for i in range(30):
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
        test_window = Window(candles=candles)

        # Mock to return our test window
        with mock.patch.object(WindowDataset, "_load_window") as mock_load:
            dataset = WindowDataset(**dataset_params)
            dataset.window = test_window  # Set our test window

            # Test getting a window by a timestamp in the middle
            target_time = base_time + timedelta(hours=15)  # Should be 16th candle
            result_window = dataset[target_time]

            # Check result is a Window instance with 10 candles
            # (default window_size in dataset_params)
            assert isinstance(result_window, Window)
            assert len(result_window) == 10

            # Check first candle (index 6) and last candle (index 15)
            assert result_window[0].time == base_time + timedelta(hours=6)
            assert result_window[-1].time == base_time + timedelta(hours=15)

            # Test with time beyond the end of data
            future_time = base_time + timedelta(hours=100)
            future_result = dataset[future_time]

            # Should return last 10 candles
            assert len(future_result) == 10
            assert future_result[0].time == base_time + timedelta(hours=20)
            assert future_result[-1].time == base_time + timedelta(hours=29)

            # Test with time before the first candle
            early_time = base_time - timedelta(hours=5)

            # Should return an empty window or the first window_size candles
            # Let's check it returns the first window_size candles
            early_result = dataset[early_time]
            assert len(early_result) == 10
            assert early_result[0].time == base_time
            assert early_result[-1].time == base_time + timedelta(hours=9)

    def test_getitem_partial_window(self, mock_from_db, dataset_params, sample_window):
        """Test __getitem__ method with insufficient data for a full window."""
        # Create a window with only 5 candles
        candles = []
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        for i in range(5):  # Only 5 candles, less than window_size=10
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
        test_window = Window(candles=candles)

        # Mock to return our test window
        with mock.patch.object(WindowDataset, "_load_window") as mock_load:
            dataset = WindowDataset(**dataset_params)
            dataset.window = test_window  # Set our test window

            # Request a window by timestamp
            target_time = base_time + timedelta(hours=3)

            # Should log a warning about partial window
            with mock.patch("pytrad.dataset.logger.warning") as mock_warning:
                result_window = dataset[target_time]

                # Check warning was logged
                assert mock_warning.call_count > 0
                assert "partial window" in str(mock_warning.call_args[0][0]).lower()

            # Should return all available candles up to target_time
            assert len(result_window) == 4
            assert result_window[0].time == base_time
            assert result_window[-1].time == base_time + timedelta(hours=3)


if __name__ == "__main__":
    """
    Main entry point for running the tests directly.
    """
    pytest.main(["-xvs", "tests/unit/test_dataset.py"])
