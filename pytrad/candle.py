from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytrad.db import MongoDBHandler
from pytrad.indicator import EMA, MA, MACD, RSI, BollingerBands, Indicator

# Evitar importación circular pero permitir tipado
if TYPE_CHECKING:
    import torch


import math

import torch


class Candle:
    def __init__(
        self,
        time: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timeframe: str,
        asset: Optional[str] = None,
        market: Optional[str] = None,
    ):
        self.time = time
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timeframe = timeframe
        self.asset = asset
        self.market = market

    def __repr__(self):
        return (
            f"Candle(time={self.time}, open={self.open}, high={self.high}, "
            f"low={self.low}, close={self.close}, volume={self.volume}, "
            f"asset={self.asset}, market={self.market})"
        )

    def is_valid(self) -> bool:
        return (
            self.low <= self.open <= self.high and self.low <= self.close <= self.high
        )

    @classmethod
    def from_document(cls, doc: dict) -> "Candle":
        return cls(
            time=doc["time"],
            open_price=doc["open"],
            high=doc["high"],
            low=doc["low"],
            close=doc["close"],
            volume=doc["volume"],
            timeframe=doc["timeframe"],
            asset=doc["asset"],
            market=doc["market"],
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> "Candle":
        return cls(
            time=series["time"],
            open_price=series["open"],
            high=series["high"],
            low=series["low"],
            close=series["close"],
            volume=series["volume"],
            timeframe=series["timeframe"],
            asset=series["asset"],
            market=series["market"],
        )


class Window:
    def __init__(
        self,
        candles: List[Candle],
    ):
        self.candles = candles

        self.computed_indicators = {}
        self.indicators = {}

        self.validate()

        self.asset = self.candles[0].asset
        self.market = self.candles[0].market
        self.timeframe = self.candles[0].timeframe

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Window":
        return cls(
            candles=[Candle.from_series(row) for index, row in df.iterrows()],
        )

    @classmethod
    def from_dict(cls, window_dict: dict) -> "Window":
        return cls(
            candles=[Candle.from_series(row) for index, row in window_dict.items()],
        )

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
        total = 5
        for indicator_name, values in self.computed_indicators.items():
            if isinstance(values, dict):
                total += len(values)
            else:
                total += 1

        return total

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
            return Window(self.candles[index])
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
                        window_dict[f"{indicator_name}_{component_name}"] = (
                            component_values
                        )
                else:
                    # Handle simple indicators
                    window_dict[indicator_name] = values

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
        self.indicators[indicator.name] = indicator.compute(self.to_df())

    def add_indicators(self, indicators: List[Indicator]):
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
                - 'market': Tensor with market identifier (scalar tensor)
                - 'asset': Tensor with asset identifier (scalar tensor)
                - 'timeframe': Tensor with timeframe identifier (scalar tensor)
        """

        # Get window data from to_dict (which already calculates returns)
        window_dict = self.to_dict()

        # Create tensors for each component
        ohlc_tensor = self._create_ohlc_tensor(window_dict)
        ohlc_returns_tensor = self._create_ohlc_returns_tensor(window_dict)
        indicators_tensor = self._create_indicators_tensor(window_dict)
        time_tensor = self._create_time_feature_tensor(self.candles)

        # Create scalar tensors for market, asset, and timeframe
        # For string data, store it directly rather than as tensors
        # This avoids compatibility issues with PyTorch string tensors
        market_value = self.market if self.market is not None else ""
        asset_value = self.asset if self.asset is not None else ""
        timeframe_value = self.timeframe if self.timeframe is not None else ""

        return {
            "ohlc": ohlc_tensor,
            "ohlc_returns": ohlc_returns_tensor,
            "indicators": indicators_tensor,
            "time_transformed": time_tensor,
            "market": market_value,
            "asset": asset_value,
            "timeframe": timeframe_value,
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

        indicator_columns = [
            col for col in window_dict.keys() if col not in basic_columns
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

    def split_by_time(self, start_date: datetime, end_date: datetime) -> "Window":
        """
        Returns a new Window containing only candles between start_date and end_date.

        Args:
            start_date: Start datetime to filter from
            end_date: End datetime to filter to

        Returns:
            Window: New window containing filtered candles
        """
        # Find indices of candles within the date range
        start_idx = None
        end_idx = None

        for i, candle in enumerate(self.candles):
            if start_idx is None and candle.time >= start_date:
                start_idx = i
            if end_idx is None and candle.time > end_date:
                end_idx = i
                break

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.candles)

        # Use slice indexing to create new Window
        return self[start_idx:end_idx]

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

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=ma_values,
                        mode="lines",
                        name=f"{ind.name}({ind.period})",
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

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=bb_values["upper_band"],
                mode="lines",
                name=f"BB Upper({ind.period}, {ind.std_dev})",
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
                name=f"BB Middle({ind.period})",
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
                name=f"BB Lower({ind.period}, {ind.std_dev})",
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

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=rsi_values,
                mode="lines",
                name=f"RSI({ind.period})",
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

        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=macd_values["macd_line"],
                mode="lines",
                name=f"MACD({ind.period},{ind.long_period})",
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
                name=f"Signal({ind.signal_period})",
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
                name="Histogram",
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
