"""
Unit tests for the Candle class of the PyTrad framework.

This test suite provides comprehensive coverage of the Candle class,
which represents a single candlestick in financial market data.

The tests verify:
1. Correct initialization with different parameters
2. Proper string representation
3. Validation logic for candle data
4. Factory methods for creating Candles from different data sources
5. Handling of edge cases

To run all tests:
    pytest -xvs tests/unit/test_candle.py

To run a specific test class:
    pytest -xvs tests/unit/test_candle.py::TestCandle

To run a specific test:
    pytest -xvs tests/unit/test_candle.py::TestCandle::test_initialization
"""

from datetime import datetime

import pandas as pd
import pytest

from pytrad.candle import Candle


class TestCandle:
    """Test suite for the Candle class."""

    @pytest.fixture
    def sample_candle(self):
        """Create a sample candle for testing."""
        return Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
            asset="EURUSD",
            market="forex",
        )

    @pytest.fixture
    def sample_document(self):
        """Create a sample document dictionary for testing."""
        return {
            "time": datetime(2023, 1, 1, 12, 0, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
            "timeframe": "1h",
            "asset": "EURUSD",
            "market": "forex",
        }

    @pytest.fixture
    def sample_series(self):
        """Create a sample pandas Series for testing."""
        return pd.Series(
            {
                "time": datetime(2023, 1, 1, 12, 0, 0),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
                "timeframe": "1h",
                "asset": "EURUSD",
                "market": "forex",
            }
        )

    def test_initialization(self, sample_candle):
        """Test that Candle initializes correctly with parameters."""
        # Check that all attributes are set correctly
        assert sample_candle.time == datetime(2023, 1, 1, 12, 0, 0)
        assert sample_candle.open == 100.0
        assert sample_candle.high == 105.0
        assert sample_candle.low == 95.0
        assert sample_candle.close == 102.0
        assert sample_candle.volume == 1000.0
        assert sample_candle.timeframe == "1h"
        assert sample_candle.asset == "EURUSD"
        assert sample_candle.market == "forex"

    def test_initialization_with_optional_fields(self):
        """Test initialization with optional fields as None."""
        candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
        )

        assert candle.asset is None
        assert candle.market is None

    def test_repr(self, sample_candle):
        """Test the string representation of Candle."""
        repr_str = str(sample_candle)

        # Verify that the string contains all important attributes
        assert "Candle" in repr_str
        assert "time=" in repr_str
        assert "open=" in repr_str
        assert "high=" in repr_str
        assert "low=" in repr_str
        assert "close=" in repr_str
        assert "volume=" in repr_str
        assert "asset=" in repr_str
        assert "market=" in repr_str

        # Verify that attribute values are included
        assert "100.0" in repr_str  # open
        assert "105.0" in repr_str  # high
        assert "95.0" in repr_str  # low
        assert "102.0" in repr_str  # close
        assert "1000.0" in repr_str  # volume
        assert "EURUSD" in repr_str  # asset
        assert "forex" in repr_str  # market

    def test_is_valid_with_valid_candle(self, sample_candle):
        """Test is_valid returns True for a valid candle."""
        assert sample_candle.is_valid() is True

    def test_is_valid_with_invalid_candle(self):
        """Test is_valid returns False for invalid candles."""
        # Case 1: open < low
        invalid_candle1 = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=90.0,  # Open is less than low
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
        )
        assert invalid_candle1.is_valid() is False

        # Case 2: open > high
        invalid_candle2 = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=110.0,  # Open is greater than high
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            timeframe="1h",
        )
        assert invalid_candle2.is_valid() is False

        # Case 3: close < low
        invalid_candle3 = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=90.0,  # Close is less than low
            volume=1000.0,
            timeframe="1h",
        )
        assert invalid_candle3.is_valid() is False

        # Case 4: close > high
        invalid_candle4 = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=110.0,  # Close is greater than high
            volume=1000.0,
            timeframe="1h",
        )
        assert invalid_candle4.is_valid() is False

    def test_from_document(self, sample_document):
        """Test creating a Candle from a document dictionary."""
        candle = Candle.from_document(sample_document)

        # Check that all attributes match the document
        assert candle.time == sample_document["time"]
        assert candle.open == sample_document["open"]
        assert candle.high == sample_document["high"]
        assert candle.low == sample_document["low"]
        assert candle.close == sample_document["close"]
        assert candle.volume == sample_document["volume"]
        assert candle.timeframe == sample_document["timeframe"]
        assert candle.asset == sample_document["asset"]
        assert candle.market == sample_document["market"]

    def test_from_series(self, sample_series):
        """Test creating a Candle from a pandas Series."""
        candle = Candle.from_series(sample_series)

        # Check that all attributes match the series
        assert candle.time == sample_series["time"]
        assert candle.open == sample_series["open"]
        assert candle.high == sample_series["high"]
        assert candle.low == sample_series["low"]
        assert candle.close == sample_series["close"]
        assert candle.volume == sample_series["volume"]
        assert candle.timeframe == sample_series["timeframe"]
        assert candle.asset == sample_series["asset"]
        assert candle.market == sample_series["market"]

    def test_with_extreme_values(self):
        """Test Candle with extreme numerical values."""
        # Very large values
        large_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=1e9,
            high=1e9 + 1000,
            low=1e9 - 1000,
            close=1e9 + 500,
            volume=1e12,
            timeframe="1h",
        )
        assert large_candle.is_valid() is True

        # Very small values
        small_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=1e-9,
            high=1e-9 + 1e-10,
            low=1e-9 - 1e-10,
            close=1e-9 + 5e-11,
            volume=1e-12,
            timeframe="1h",
        )
        assert small_candle.is_valid() is True

        # Zero values (edge case but still valid)
        zero_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
            timeframe="1h",
        )
        assert zero_candle.is_valid() is True

    def test_with_different_timeframes(self):
        """Test Candle with different timeframe values."""
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

        for tf in timeframes:
            candle = Candle(
                time=datetime(2023, 1, 1, 12, 0, 0),
                open_price=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                timeframe=tf,
            )
            assert candle.timeframe == tf

    def test_equal_prices(self):
        """Test Candle with equal prices (for example in a doji pattern)."""
        # All prices equal
        doji_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000.0,
            timeframe="1h",
        )
        assert doji_candle.is_valid() is True

        # High equals low (zero range)
        zero_range_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000.0,
            timeframe="1h",
        )
        assert zero_range_candle.is_valid() is True

    def test_negative_prices(self):
        """Test Candle with negative prices (unusual but possible in some derivatives)."""
        negative_candle = Candle(
            time=datetime(2023, 1, 1, 12, 0, 0),
            open_price=-100.0,
            high=-95.0,
            low=-105.0,
            close=-98.0,
            volume=1000.0,
            timeframe="1h",
        )
        assert negative_candle.is_valid() is True


if __name__ == "__main__":
    """
    Main entry point for running the tests directly.
    """
    pytest.main(["-xvs", "tests/unit/test_candle.py"])
