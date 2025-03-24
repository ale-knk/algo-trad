from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytrad.candle import Candle
from pytrad.db import MongoDBHandler
from pytrad.indexer import AssetIndexer, MarketIndexer, TimeframeIndexer
from pytrad.indicator import EMA, MA, MACD, RSI, BollingerBands, Indicator
from pytrad.indicator_name import IndicatorName
from pytrad.normalizer import FeatureNormalizer

# Evitar importación circular pero permitir tipado
if TYPE_CHECKING:
    import torch


import math

import torch


class Window:
    def __init__(
        self,
        candles: List[Candle],
        indicators: Optional[Dict[Union[str, IndicatorName], Any]] = None,
    ):
        self.candles = candles

        # Convert string indicator keys to IndicatorName objects if necessary
        self.indicators = {}
        if indicators is not None:
            for key, value in indicators.items():
                if isinstance(key, str):
                    indicator_name = IndicatorName.from_string(key).to_string()
                    self.indicators[indicator_name] = value
                else:
                    self.indicators[key] = value

        self.validate()

        self.start_date = self.candles[0].time
        self.end_date = self.candles[-1].time

        # Initialize indexers for encoding asset, market, and timeframe
        self.asset_indexer = AssetIndexer()
        self.market_indexer = MarketIndexer()
        self.timeframe_indexer = TimeframeIndexer()

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Window":
        # Extract indicator columns from the DataFrame
        indicators = {}

        # Basic columns that are not indicators
        basic_columns = [
            "time",
            "timeframe",
            "market",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_ret",
            "high_ret",
            "low_ret",
            "close_ret",
        ]

        # Find indicator columns and group them by indicator type
        indicator_columns = [col for col in df.columns if col not in basic_columns]

        # Process indicator columns
        for col in indicator_columns:
            # Parse the column name into an IndicatorName object
            indicator_name = IndicatorName.from_string(col)

            # If this is a complex indicator with a component
            if indicator_name.component:
                # Get the base indicator name without component
                base_indicator = indicator_name.without_component().to_string()

                # Initialize the complex indicator dictionary if needed
                if base_indicator not in indicators:
                    indicators[base_indicator] = {}

                # Add the component values
                indicators[base_indicator][indicator_name.component] = df[col].tolist()
            else:
                # This is a simple indicator (single value series)
                indicators[indicator_name.to_string()] = df[col].tolist()

        # Create a new window instance with candles and extracted indicators
        window = cls(
            candles=[Candle.from_series(row) for index, row in df.iterrows()],
            indicators=indicators,
        )

        return window

    @classmethod
    def from_db(
        cls, asset: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> "Window":
        mongodb_handler = MongoDBHandler()
        mongodb_handler.set_collection(asset)

        query = {
            "timeframe": timeframe,
            "time": {"$gte": start_date, "$lte": end_date},
        }

        cursor = mongodb_handler.collection.find(query).sort("time", 1)
        candles = []
        for doc in cursor:
            doc.pop("_id", None)
            candles.append(Candle.from_document(doc))

        return cls(candles=candles)

    @property
    def n_features(self) -> int:
        total = 10
        for indicator_name, values in self.indicators.items():
            if isinstance(values, dict):
                total += len(values)
            else:
                total += 1

        return total

    @property
    def asset(self) -> str:
        if not self.candles:
            return ""
        return self.candles[0].asset or ""

    @property
    def market(self) -> str:
        if not self.candles:
            return ""
        return self.candles[0].market or ""

    @property
    def timeframe(self) -> str:
        if not self.candles:
            return ""
        return self.candles[0].timeframe or ""

    def validate(self):
        market_set, asset_set, timeframe_set = set(), set(), set()
        for candle in self.candles:
            if candle.market is None or candle.asset is None:
                raise ValueError("All candles must have a market and asset feature.")
            market_set.add(candle.market)
            asset_set.add(candle.asset)
            timeframe_set.add(candle.timeframe)

        if len(market_set) > 1:
            raise ValueError("All candles must have the same market feature.")
        if len(asset_set) > 1:
            raise ValueError("All candles must have the same asset feature.")
        if len(timeframe_set) > 1:
            raise ValueError("All candles must have the same timeframe feature.")

    def __len__(self) -> int:
        return len(self.candles)

    def __getitem__(self, index):
        if isinstance(index, slice):
            window = Window(self.candles[index])

            # Apply the same slice to the indicators
            for indicator_name, indicator_values in self.indicators.items():
                if isinstance(indicator_values, dict):
                    # For complex indicators (like MACD)
                    window.indicators[indicator_name] = {
                        component: values[index]
                        for component, values in indicator_values.items()
                    }
                else:
                    # For simple indicators
                    window.indicators[indicator_name] = indicator_values[index]

            return window
        elif isinstance(index, int):
            return self.candles[index]
        else:
            raise TypeError("Invalid argument type.")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if index < 0:
                index += len(self.candles)
            if index < 0 or index >= len(self.candles):
                raise IndexError("Index out of range.")
            if not isinstance(value, Candle):
                raise TypeError("Value must be a Candle instance.")
            self.candles[index] = value
        else:
            raise TypeError("Invalid argument type.")

    def _copy_indicators_to_window(self, window, index):
        """
        Copy indicators from this window to the new sliced window.

        Args:
            window: The target window to copy indicators to
            index: The slice index used to create the new window
        """
        if not self.indicators:
            return

        # Determine the start and end index of the slice
        start, stop, step = self._normalize_slice_indices(index)

        # Copy indicators to the new window
        for indicator_name, indicator_values in self.indicators.items():
            if isinstance(indicator_values, dict):
                self._copy_complex_indicator(
                    window, indicator_name, indicator_values, start, stop, step
                )
            else:
                self._copy_simple_indicator(
                    window, indicator_name, indicator_values, start, stop, step
                )

        # Also update indicators
        if self.indicators:
            window.indicators = self.indicators.copy()

    def _normalize_slice_indices(self, index):
        """
        Convert slice indices to normalized positive values.

        Args:
            index: The slice object

        Returns:
            tuple: (start, stop, step) indices
        """
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else len(self.candles)
        step = index.step if index.step is not None else 1

        # Ensure indices are positive
        if start < 0:
            start += len(self.candles)
        if stop < 0:
            stop += len(self.candles)

        return start, stop, step

    def _copy_complex_indicator(
        self, window, indicator_name, indicator_values, start, stop, step
    ):
        """
        Copy a complex indicator (dictionary of values) to the target window.

        Args:
            window: The target window
            indicator_name: Name of the indicator
            indicator_values: Dictionary of indicator component values
            start, stop, step: Normalized slice indices
        """
        window.indicators[indicator_name] = {}

        for component_name, component_values in indicator_values.items():
            # Calculate how many values to skip from the beginning
            # This handles indicators that might have fewer values than candles (like moving averages)
            adjusted_start, adjusted_stop = self._calculate_adjusted_indices(
                component_values, start, stop
            )

            if adjusted_start < len(component_values):
                # Extract the slice of indicator values
                sliced_values = component_values[adjusted_start:adjusted_stop:step]
                window.indicators[indicator_name][component_name] = sliced_values

    def _copy_simple_indicator(
        self, window, indicator_name, indicator_values, start, stop, step
    ):
        """
        Copy a simple indicator (list of values) to the target window.

        Args:
            window: The target window
            indicator_name: Name of the indicator
            indicator_values: List of indicator values
            start, stop, step: Normalized slice indices
        """
        adjusted_start, adjusted_stop = self._calculate_adjusted_indices(
            indicator_values, start, stop
        )

        if adjusted_start < len(indicator_values):
            sliced_values = indicator_values[adjusted_start:adjusted_stop:step]
            window.indicators[indicator_name] = sliced_values

    def _calculate_adjusted_indices(self, values, start, stop):
        """
        Calculate adjusted start and stop indices for indicator values.

        This handles cases where indicators might have fewer values than candles.

        Args:
            values: The indicator values list
            start, stop: Original slice indices

        Returns:
            tuple: (adjusted_start, adjusted_stop)
        """
        values_to_skip = max(0, len(self.candles) - len(values))
        adjusted_start = max(0, start - values_to_skip)
        adjusted_stop = max(0, stop - values_to_skip)

        return adjusted_start, adjusted_stop

    def clear_indicators_nans(self, inplace=True):
        """Remove initial rows with NaN values from indicators."""
        df = self.to_df()

        # Find the first row where all values are valid (no NaNs)
        first_valid_idx = 0
        for idx, row in df.iterrows():
            if not row.isna().any():
                # Found the first row with no NaNs
                loc = df.index.get_loc(idx)
                # Handle the case where loc might be a slice or array
                if isinstance(loc, (slice, np.ndarray)):
                    if isinstance(loc, slice):
                        first_valid_idx = loc.start if loc.start is not None else 0
                    else:  # ndarray
                        first_valid_idx = int(loc[0]) if len(loc) > 0 else 0
                else:
                    first_valid_idx = int(loc)
                break

        if first_valid_idx > 0:
            if inplace:
                # Update the current instance's attributes
                self.candles = self.candles[first_valid_idx:]
                # Update indicators by removing the same number of initial values
                for indicator_name, values in self.indicators.items():
                    if isinstance(values, dict):
                        # For complex indicators (like MACD)
                        for component_name, component_values in values.items():
                            self.indicators[indicator_name][component_name] = (
                                component_values[first_valid_idx:]
                            )
                    else:
                        # For simple indicators
                        self.indicators[indicator_name] = values[first_valid_idx:]
                return self
            else:
                # Return new window without the NaN rows
                return self[first_valid_idx:]
        return self

    def normalize(self, normalizer: FeatureNormalizer, inplace=True):
        if normalizer is None:
            return self

        # If normalizer is not fitted, fit it
        if not normalizer.is_fitted:
            normalizer.fit(self.to_df())

        # Transform the data
        df_to_normalize = self.to_df()
        normalized_df = normalizer.transform(df_to_normalize)

        if inplace:
            # Create a new window from the normalized DataFrame
            normalized_window = self.from_df(normalized_df)

            # Update the current instance's attributes
            self.candles = normalized_window.candles
            self.indicators = normalized_window.indicators
            return self
        else:
            # Return a new Window instance
            return self.from_df(normalized_df)

    def get_subwindow_by_datetime(self, target_datetime, window_size):
        """
        Returns a window ending at the last candle before the given datetime.

        Args:
            target_datetime: datetime to find the window before
            window_size: size of the window to return

        Returns:
            Window: A window of size window_size ending before target_datetime
        """
        # Find index of last candle before target_datetime
        end_idx = None
        for i, candle in enumerate(self.candles):
            if candle.time > target_datetime:
                end_idx = i - 1
                break

        if end_idx is None:
            end_idx = len(self.candles) - 1

        # Calculate start index based on window size
        start_idx = max(0, end_idx - window_size + 1)
        if end_idx - start_idx + 1 < window_size:
            raise ValueError(
                f"Insufficient data for window at {target_datetime}. Need {window_size} candles but only have {end_idx - start_idx + 1}."
            )

        # Return window slice
        return self[start_idx : end_idx + 1]

    def to_dict(self):
        """
        Convert window data to a dictionary format with organized structure.

        Returns:
            dict: Dictionary with the following sections:
                - Basic metadata (time, timeframe, market, asset)
                - OHLC price data
                - OHLC return data (logarithmic returns)
                - Volume data
                - Indicator data (if available)
        """
        # Initialize dictionary with basic sections
        window_dict = {
            # Metadata
            "time": [],
            "timeframe": [],
            "market": [],
            "asset": [],
            # OHLC prices
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            # Volume
            "volume": [],
            # Returns (will be calculated)
            "open_ret": [],
            "high_ret": [],
            "low_ret": [],
            "close_ret": [],
        }

        # Fill in data from candles
        for candle in self.candles:
            # Metadata
            window_dict["time"].append(candle.time)
            window_dict["timeframe"].append(candle.timeframe)
            window_dict["market"].append(candle.market)
            window_dict["asset"].append(candle.asset)

            # OHLC prices
            window_dict["open"].append(candle.open)
            window_dict["high"].append(candle.high)
            window_dict["low"].append(candle.low)
            window_dict["close"].append(candle.close)

            # Volume
            window_dict["volume"].append(candle.volume)

        # Calculate logarithmic returns for OHLC

        # Calculate returns for each price series (open, high, low, close)
        for price_type in ["open", "high", "low", "close"]:
            prices = np.array(window_dict[price_type])
            returns = np.empty(len(prices))

            # First position is 0.0 instead of NaN for numerical stability
            returns[0] = 0.0

            # Calculate log returns for the rest: log(P_t / P_{t-1})
            if len(prices) > 1:
                returns[1:] = np.log(prices[1:] / prices[:-1])

            # Add to the dictionary
            window_dict[f"{price_type}_ret"] = returns.tolist()

        # Add indicator data if available
        if self.indicators:
            for indicator_name, values in self.indicators.items():
                if isinstance(values, dict):
                    # Handle complex indicators (e.g., MACD, Bollinger Bands)
                    for component_name, component_values in values.items():
                        # Ensure indicator values match the length of the window
                        component_values_list = list(component_values)
                        if len(component_values_list) != len(self.candles):
                            # Need to pad or truncate the values to match window length
                            if len(component_values_list) < len(self.candles):
                                # Pad with NaN at the beginning
                                padding = [np.nan] * (
                                    len(self.candles) - len(component_values_list)
                                )
                                component_values_list = padding + component_values_list
                            else:
                                # Truncate to match window length (take most recent values)
                                component_values_list = component_values_list[
                                    -len(self.candles) :
                                ]

                        key = (
                            IndicatorName.from_string(indicator_name)
                            .with_component(component_name)
                            .to_string()
                        )
                        window_dict[key] = component_values_list
                else:
                    # Handle simple indicators
                    values_list = list(values)
                    if len(values_list) != len(self.candles):
                        # Need to pad or truncate the values to match window length
                        if len(values_list) < len(self.candles):
                            # Pad with NaN at the beginning
                            padding = [np.nan] * (len(self.candles) - len(values_list))
                            values_list = padding + values_list
                        else:
                            # Truncate to match window length (take most recent values)
                            values_list = values_list[-len(self.candles) :]

                    window_dict[indicator_name] = values_list

        return window_dict

    def to_df(self) -> pd.DataFrame:
        """
        Convert window to a pandas DataFrame.

        Args:
            use_log_returns: Deprecated, returns are always included now

        Returns:
            DataFrame with candle data and indicators
        """
        return pd.DataFrame(self.to_dict())

    def add_indicator(self, indicator: Indicator):
        """
        Add an indicator to this window.

        Args:
            indicator: Indicator instance to add
        """
        values = indicator.compute(self.to_df())
        self.indicators[indicator.name.to_string()] = values

    def add_indicators(self, indicators: List[Indicator]):
        """
        Add multiple indicators to this window.

        Args:
            indicators: List of Indicator instances to add
        """
        for indicator in indicators:
            self.add_indicator(indicator)

    def to_tensors(self) -> dict:
        """
        Converts the window to a dictionary of PyTorch tensors.

        Returns:
            dict: A dictionary with the following keys:
                - 'ohlc': Tensor with OHLC price data (without returns)
                - 'ohlc_returns': Tensor with log returns for OHLC
                - 'indicators': Tensor with all indicator values
                - 'time_transformed': Tensor with cyclic time encoding
                - 'market': Tensor with market identifier (tensor with encoded index)
                - 'asset': Tensor with asset identifier (tensor with encoded index)
                - 'timeframe': Tensor with timeframe identifier (tensor with encoded index)
        """

        # Get window data from to_dict (which already calculates returns)
        window_dict = self.to_dict()

        # Create tensors for each component
        ohlc_tensor = self._create_ohlc_tensor(window_dict)
        ohlc_returns_tensor = self._create_ohlc_returns_tensor(window_dict)
        indicators_tensor = self._create_indicators_tensor(window_dict)
        time_tensor = self._create_time_feature_tensor(self.candles)

        # Convert market, asset, and timeframe to tensors using indexers
        market_value = self.market if self.market is not None else ""
        asset_value = self.asset if self.asset is not None else ""
        timeframe_value = self.timeframe if self.timeframe is not None else ""

        # Create tensor indices using indexers
        market_tensor = torch.tensor(
            [self.market_indexer.encode(market_value)], dtype=torch.long
        )
        asset_tensor = torch.tensor(
            [self.asset_indexer.encode(asset_value)], dtype=torch.long
        )
        timeframe_tensor = torch.tensor(
            [self.timeframe_indexer.encode(timeframe_value)], dtype=torch.long
        )

        return {
            "ohlc": ohlc_tensor,
            "ohlc_ret": ohlc_returns_tensor,
            "indicators": indicators_tensor,
            "time": time_tensor,
            "market": market_tensor,
            "asset": asset_tensor,
            "timeframe": timeframe_tensor,
        }

    def _create_ohlc_tensor(self, window_dict: dict) -> "torch.Tensor":
        """
        Creates a tensor with OHLC price data.

        Args:
            window_dict: Dictionary from to_dict method

        Returns:
            torch.Tensor: Tensor with OHLC price data
        """
        import torch

        # Extract OHLC price data
        price_columns = ["open", "high", "low", "close"]
        price_data = []

        for i in range(len(window_dict["open"])):
            row_data = []
            for col in price_columns:
                try:
                    row_data.append(float(window_dict[col][i]))
                except (IndexError, KeyError):
                    row_data.append(0.0)
            price_data.append(row_data)

        return torch.tensor(price_data, dtype=torch.float32)

    def _create_ohlc_returns_tensor(self, window_dict: dict) -> "torch.Tensor":
        """
        Creates a tensor with log returns for OHLC.

        Args:
            window_dict: Dictionary from to_dict method

        Returns:
            torch.Tensor: Tensor with log returns
        """
        import torch

        # Extract return data
        return_columns = ["open_ret", "high_ret", "low_ret", "close_ret"]
        return_data = []

        for i in range(len(window_dict["open_ret"])):
            row_data = []
            for col in return_columns:
                try:
                    value = window_dict[col][i]
                    # Handle NaN values (first entry)
                    row_data.append(0.0 if np.isnan(value) else float(value))
                except (IndexError, KeyError):
                    row_data.append(0.0)
            return_data.append(row_data)

        return torch.tensor(return_data, dtype=torch.float32)

    def _create_indicators_tensor(self, window_dict: dict) -> "torch.Tensor":
        """
        Creates a tensor with all indicator values.

        Args:
            window_dict: Dictionary from to_dict method

        Returns:
            torch.Tensor: Tensor with indicator values
        """
        import torch

        # Get all indicator columns (any column that's not OHLC, returns, or metadata)
        basic_columns = [
            "time",
            "timeframe",
            "market",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_ret",
            "high_ret",
            "low_ret",
            "close_ret",
        ]

        # Find all indicator column names in the dict
        indicator_columns = [
            col
            for col in window_dict.keys()
            if col not in basic_columns
            and (
                "-" in col
                or any(
                    ind in col
                    for ind in [
                        "MA_",
                        "RSI_",
                        "EMA_",
                        "BB_",
                        "MACD_",
                        "ATR_",
                        "CCI_",
                        "MFI_",
                        "OBV_",
                    ]
                )
            )
        ]

        if not indicator_columns:
            # Return empty tensor if no indicators (shape: [sequence_length, 0])
            return torch.zeros((len(window_dict["open"]), 0), dtype=torch.float32)

        indicator_data = []
        for i in range(len(window_dict["open"])):
            row_data = []
            for col in indicator_columns:
                try:
                    value = window_dict[col][i]
                    # Handle NaN values
                    row_data.append(0.0 if np.isnan(value) else float(value))
                except (IndexError, KeyError):
                    row_data.append(0.0)
            indicator_data.append(row_data)

        return torch.tensor(indicator_data, dtype=torch.float32)

    def _create_time_feature_tensor(self, candles: List[Candle]) -> "torch.Tensor":
        """
        Create time feature tensor with cyclic encoding for all candles.

        Args:
            candles: List of candles

        Returns:
            torch.Tensor: Tensor with time features
        """
        import torch

        time_features = []
        for candle in candles:
            time_features.append(self._extract_time_features(candle))

        # Convert to tensor
        return torch.tensor(time_features, dtype=torch.float32)

    def _extract_time_features(self, candle):
        """
        Extract time features from a single candle with cyclic encoding.

        Args:
            candle: Candle object

        Returns:
            list: List of time features
        """

        # Basic time features
        weekday = candle.time.weekday()
        hour = candle.time.hour + candle.time.minute / 60 + candle.time.second / 3600

        # Cyclic encoding for weekday
        weekday_sin = math.sin(2 * math.pi * weekday / 7)
        weekday_cos = math.cos(2 * math.pi * weekday / 7)

        # Cyclic encoding for hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # Day of month (1-31)
        day_of_month = candle.time.day
        day_of_month_sin = math.sin(2 * math.pi * day_of_month / 31)
        day_of_month_cos = math.cos(2 * math.pi * day_of_month / 31)

        # Month of year (1-12)
        month = candle.time.month
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        # Day of year (1-366)
        day_of_year = candle.time.timetuple().tm_yday
        day_of_year_sin = math.sin(2 * math.pi * day_of_year / 366)
        day_of_year_cos = math.cos(2 * math.pi * day_of_year / 366)

        # Market sessions (one-hot encoding)
        utc_hour = candle.time.hour  # Assuming time is in UTC
        asian_session = 1.0 if 0 <= utc_hour < 8 else 0.0
        european_session = 1.0 if 7 <= utc_hour < 16 else 0.0
        us_session = 1.0 if 13 <= utc_hour < 22 else 0.0

        # Is weekend
        is_weekend = 1.0 if weekday >= 5 else 0.0  # 5=Saturday, 6=Sunday

        # Quarter of year (1-4)
        quarter = (month - 1) // 3 + 1
        quarter_sin = math.sin(2 * math.pi * quarter / 4)
        quarter_cos = math.cos(2 * math.pi * quarter / 4)

        # Hour normalized (0-1)
        hour_normalized = hour / 24

        # Minute cyclic
        minute = candle.time.minute
        minute_sin = math.sin(2 * math.pi * minute / 60)
        minute_cos = math.cos(2 * math.pi * minute / 60)

        # Combine all time features
        return [
            weekday_sin,
            weekday_cos,  # Day of week
            hour_sin,
            hour_cos,  # Hour of day
            day_of_month_sin,
            day_of_month_cos,  # Day of month
            month_sin,
            month_cos,  # Month of year
            day_of_year_sin,
            day_of_year_cos,  # Day of year
            quarter_sin,
            quarter_cos,  # Quarter of year
            minute_sin,
            minute_cos,  # Minute cyclic
            asian_session,
            european_session,
            us_session,  # Market sessions
            is_weekend,  # Weekend flag
            hour_normalized,  # Hour as normalized value
        ]

    def get_indicator_df(self):
        """
        Get DataFrame with price data for indicator calculations.

        Returns:
            pd.DataFrame: DataFrame with OHLC data
        """
        if not self.candles:
            raise ValueError("No candles available to create DataFrame.")

        data = {
            "open": [candle.open for candle in self.candles],
            "high": [candle.high for candle in self.candles],
            "low": [candle.low for candle in self.candles],
            "close": [candle.close for candle in self.candles],
            "volume": [candle.volume for candle in self.candles],
        }
        index = [candle.time for candle in self.candles]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        df.index.name = "time"
        return df

    def plot(self, indicators=None):
        """
        Plot candlestick chart with volume and optional indicators.

        Args:
            indicators: List of indicator objects to include in the plot
        """
        if not self.candles:
            raise ValueError("No candles to plot.")

        # Create figure with subplots
        fig = self._create_plot_figure(indicators)

        # Show the figure
        fig.show()

    def _create_plot_figure(self, indicators=None):
        """
        Create a plotly figure with candlestick chart and indicators.

        Args:
            indicators: List of indicator objects to include in the plot

        Returns:
            go.Figure: Plotly figure object
        """
        # Create DataFrame for plotting
        df = self._create_plot_dataframe()

        # Organize indicators into overlay and separate types
        overlay_indicators, separate_indicators = self._organize_indicators(indicators)

        # Determine layout based on number of indicators
        num_rows = 2 + len(
            separate_indicators
        )  # Candles + Volume + Separate indicators
        row_heights = [0.5] + [0.2] * (num_rows - 1)

        # Create subplots
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )

        # Add candlestick chart
        self._add_candlestick_to_figure(fig, df)

        # Add volume chart
        self._add_volume_to_figure(fig, df)

        # Add indicators
        indicator_df = self.get_indicator_df()
        current_row = 3

        # Add overlay indicators to candlestick chart
        self._add_overlay_indicators(fig, indicator_df, overlay_indicators, df)

        # Add separate indicators on their own subplots
        current_row = self._add_separate_indicators(
            fig, indicator_df, separate_indicators, df, current_row
        )

        # Update layout
        self._update_figure_layout(fig, num_rows)

        return fig

    def _create_plot_dataframe(self):
        """
        Create a DataFrame for plotting from candles.

        Returns:
            pd.DataFrame: DataFrame with OHLC data formatted for plotting
        """
        data = {
            "Open": [candle.open for candle in self.candles],
            "High": [candle.high for candle in self.candles],
            "Low": [candle.low for candle in self.candles],
            "Close": [candle.close for candle in self.candles],
            "Volume": [candle.volume for candle in self.candles],
        }
        index = [candle.time for candle in self.candles]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        df.index.name = "time"
        return df

    def _organize_indicators(self, indicators):
        """
        Organize indicators into overlay and separate types.

        Args:
            indicators: List of indicator objects

        Returns:
            tuple: (overlay_indicators, separate_indicators)
        """
        overlay_indicators = []
        separate_indicators = []

        if indicators:
            for ind in indicators:
                if isinstance(ind, (MA, EMA)) or (
                    hasattr(ind, "plot_overlay") and ind.plot_overlay
                ):
                    overlay_indicators.append(ind)
                else:
                    separate_indicators.append(ind)

        return overlay_indicators, separate_indicators

    def _add_candlestick_to_figure(self, fig, df):
        """
        Add candlestick chart to the figure.

        Args:
            fig: Plotly figure object
            df: DataFrame with OHLC data
        """
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candlestick",
            ),
            row=1,
            col=1,
        )

    def _add_volume_to_figure(self, fig, df):
        """
        Add volume chart to the figure.

        Args:
            fig: Plotly figure object
            df: DataFrame with volume data
        """
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(0, 0, 255, 0.5)",
            ),
            row=2,
            col=1,
        )

    def _add_overlay_indicators(self, fig, indicator_df, overlay_indicators, df):
        """
        Add overlay indicators to the candlestick chart.

        Args:
            fig: Plotly figure object
            indicator_df: DataFrame for indicator calculations
            overlay_indicators: List of overlay indicators
            df: DataFrame with OHLC data
        """
        for ind in overlay_indicators:
            if isinstance(ind, (MA, EMA)):
                # Simple moving averages
                ma_values = ind.compute(indicator_df)

                # Adjust the length difference (MAs only have values starting from N)
                x_values = df.index[-(len(ma_values)) :]

                # Create a readable name for the legend
                ind_name = ind.name.to_string()

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=ma_values,
                        mode="lines",
                        name=ind_name,
                        line=dict(width=1.5),
                    ),
                    row=1,
                    col=1,
                )
            elif isinstance(ind, BollingerBands):
                self._add_bollinger_bands(fig, ind, indicator_df, df)

    def _add_bollinger_bands(self, fig, ind, indicator_df, df):
        """
        Add Bollinger Bands to the figure.

        Args:
            fig: Plotly figure object
            ind: BollingerBands indicator
            indicator_df: DataFrame for indicator calculations
            df: DataFrame with OHLC data
        """
        bb_values = ind.compute(indicator_df)
        # Adjust for the length difference
        x_values = df.index[-(len(bb_values["middle_band"])) :]

        # Get the indicator name as string
        ind_name = ind.name.to_string()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=bb_values["upper_band"],
                mode="lines",
                name=f"{ind_name}-upper_band",
                line=dict(width=1, color="rgba(250, 128, 114, 0.7)"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=bb_values["middle_band"],
                mode="lines",
                name=f"{ind_name}-middle_band",
                line=dict(width=1, color="rgba(128, 128, 128, 0.7)"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=bb_values["lower_band"],
                mode="lines",
                name=f"{ind_name}-lower_band",
                line=dict(width=1, color="rgba(173, 216, 230, 0.7)"),
            ),
            row=1,
            col=1,
        )

    def _add_separate_indicators(
        self, fig, indicator_df, separate_indicators, df, current_row
    ):
        """
        Add separate indicators below the main chart.

        Args:
            fig: Plotly figure object
            indicator_df: DataFrame for indicator calculations
            separate_indicators: List of separate indicators
            df: DataFrame with OHLC data
            current_row: Current row index

        Returns:
            int: Updated current row index
        """
        for ind in separate_indicators:
            if isinstance(ind, RSI):
                current_row = self._add_rsi(fig, ind, indicator_df, df, current_row)
            elif isinstance(ind, MACD):
                current_row = self._add_macd(fig, ind, indicator_df, df, current_row)

        return current_row

    def _add_rsi(self, fig, ind, indicator_df, df, current_row):
        """
        Add RSI indicator to the figure.

        Args:
            fig: Plotly figure object
            ind: RSI indicator
            indicator_df: DataFrame for indicator calculations
            df: DataFrame with OHLC data
            current_row: Current row index

        Returns:
            int: Updated current row index
        """
        rsi_values = ind.compute(indicator_df)
        # Adjust for the length difference
        x_values = df.index[-(len(rsi_values)) :]

        # Get the indicator name as string
        ind_name = ind.name.to_string()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=rsi_values,
                mode="lines",
                name=ind_name,
                line=dict(width=1.5, color="purple"),
            ),
            row=current_row,
            col=1,
        )

        # Add horizontal lines at 30 and 70
        fig.add_shape(
            type="line",
            line=dict(dash="dash", width=1, color="red"),
            x0=x_values[0],
            y0=70,
            x1=x_values[-1],
            y1=70,
            row=current_row,
            col=1,
        )

        fig.add_shape(
            type="line",
            line=dict(dash="dash", width=1, color="green"),
            x0=x_values[0],
            y0=30,
            x1=x_values[-1],
            y1=30,
            row=current_row,
            col=1,
        )

        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        return current_row + 1

    def _add_macd(self, fig, ind, indicator_df, df, current_row):
        """
        Add MACD indicator to the figure.

        Args:
            fig: Plotly figure object
            ind: MACD indicator
            indicator_df: DataFrame for indicator calculations
            df: DataFrame with OHLC data
            current_row: Current row index

        Returns:
            int: Updated current row index
        """
        macd_values = ind.compute(indicator_df)
        # Adjust for the length difference
        x_values = df.index[-(len(macd_values["macd_line"])) :]

        # Get the indicator name as string
        ind_name = ind.name.to_string()

        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=macd_values["macd_line"],
                mode="lines",
                name=f"{ind_name}-macd_line",
                line=dict(color="blue"),
            ),
            row=current_row,
            col=1,
        )

        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=macd_values["signal_line"],
                mode="lines",
                name=f"{ind_name}-signal_line",
                line=dict(color="red"),
            ),
            row=current_row,
            col=1,
        )

        # Histogram
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=macd_values["macd_histogram"],
                name=f"{ind_name}-histogram",
                marker=dict(
                    color=[
                        "rgba(0,255,0,0.7)" if val >= 0 else "rgba(255,0,0,0.7)"
                        for val in macd_values["macd_histogram"]
                    ]
                ),
            ),
            row=current_row,
            col=1,
        )

        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        return current_row + 1

    def _update_figure_layout(self, fig, num_rows):
        """
        Update the layout of the figure.

        Args:
            fig: Plotly figure object
            num_rows: Number of subplot rows
        """
        fig.update_layout(
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=250 + 200 * (num_rows - 1),
            legend=dict(
                orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5
            ),
            margin=dict(t=80),
        )

        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
