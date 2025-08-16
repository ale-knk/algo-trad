from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from pytrad.indicator_name import IndicatorName


class Indicator(ABC):
    def __init__(self, period: int, use_returns: bool = False):
        self.period = period
        self.use_returns = use_returns

    @property
    def name(self) -> IndicatorName:
        """
        Returns the IndicatorName object representing this indicator.
        Can be overridden in subclasses to provide more specific names.
        """
        return IndicatorName(
            base_name=self.__class__.__name__,
            parameters=[self.period],
            uses_returns=self.use_returns,
        )

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[List[float], Dict[str, List[float]]]:
        """
        Compute the indicator values based on the input DataFrame.

        Args:
            data: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
                 If use_returns=True, expected columns are 'open_ret', 'high_ret', 'low_ret', 'close_ret'

        Returns:
            Either a list of float values or a dictionary of lists
        """
        pass


class ATR(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period + 1:
            raise ValueError("Not enough data to calculate ATR")

        # Select correct columns based on use_returns
        if self.use_returns:
            high = data["high_returns"].astype(float).values
            low = data["low_returns"].astype(float).values
            close = data["close_returns"].astype(float).values
        else:
            high = data["high"].astype(float).values
            low = data["low"].astype(float).values
            close = data["close"].astype(float).values

        tr = [
            max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            for i in range(1, len(close))
        ]

        # Wilder smoothing: first ATR is simple average
        atr = [float(np.mean(tr[: self.period]))]
        for i in range(self.period, len(tr)):
            atr.append(float((atr[-1] * (self.period - 1) + tr[i]) / self.period))

        # Make sure all values are Python floats
        return [float(x) for x in atr]


class BollingerBands(Indicator):
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2,
        use_returns: bool = False,
    ):
        super().__init__(period, use_returns)
        self.std_dev = std_dev

    @property
    def name(self) -> IndicatorName:
        return IndicatorName(
            base_name="BB",
            parameters=[self.period, self.std_dev],
            uses_returns=self.use_returns,
        )

    def compute(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate Bollinger Bands.")

        # Select the appropriate column based on use_returns
        if self.use_returns:
            price_data = np.array(data["close_returns"].values, dtype=float)
        else:
            price_data = np.array(data["close"].values, dtype=float)

        # Check if we have enough data
        if len(price_data) < self.period:
            raise ValueError("Not enough data to calculate Bollinger Bands")

        sma = np.convolve(price_data, np.ones(self.period) / self.period, mode="valid")
        std_dev = np.array(
            [np.std(price_data[i : i + self.period]) for i in range(len(sma))]
        )
        upper_band = sma + self.std_dev * std_dev
        lower_band = sma - self.std_dev * std_dev

        return {
            "lower_band": lower_band.tolist(),
            "middle_band": sma.tolist(),
            "upper_band": upper_band.tolist(),
        }


class CCI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate CCI")

        # Select correct columns based on use_returns
        if self.use_returns:
            high = data["high_returns"].astype(float).values
            low = data["low_returns"].astype(float).values
            close = data["close_returns"].astype(float).values
        else:
            high = data["high"].astype(float).values
            low = data["low"].astype(float).values
            close = data["close"].astype(float).values

        # Convert arrays to numpy arrays with explicit float dtype
        high_np = np.array(high, dtype=float)
        low_np = np.array(low, dtype=float)
        close_np = np.array(close, dtype=float)

        # Calculate typical price
        typical_price = (high_np + low_np + close_np) / 3.0

        cci = []
        for i in range(self.period - 1, len(typical_price)):
            tp_slice = typical_price[i - self.period + 1 : i + 1]
            sma = np.mean(tp_slice)
            mean_deviation = np.mean(np.abs(tp_slice - sma))
            if mean_deviation == 0:
                cci_value = 0.0
            else:
                cci_value = (typical_price[i] - sma) / (0.015 * mean_deviation)
            cci.append(float(cci_value))

        return cci


class EMA(Indicator):
    def __init__(self, period: int, use_returns: bool = False):
        super().__init__(period, use_returns)

    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate exponential moving average")

        # Select the appropriate column based on use_returns
        if self.use_returns:
            price_data = np.array(data["close_returns"].values, dtype=float)
        else:
            price_data = np.array(data["close"].values, dtype=float)

        # Check if we have enough data
        if len(price_data) < self.period:
            raise ValueError("Not enough data to calculate EMA")

        ema_values = self._calculate_ema(price_data)
        return ema_values.tolist()

    def _calculate_ema(self, data: np.ndarray) -> np.ndarray:
        ema = np.zeros_like(data)
        alpha = 2 / (self.period + 1)
        ema[: self.period] = np.mean(data[: self.period])
        for i in range(self.period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema  # Return full length array instead of slicing


class MA(Indicator):
    def __init__(self, period: int, use_returns: bool = False):
        super().__init__(period, use_returns)

    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate moving average")

        # Select the appropriate column based on use_returns
        if self.use_returns:
            price_data = np.array(data["close_returns"].values, dtype=float)
        else:
            price_data = np.array(data["close"].values, dtype=float)

        # Check if we have enough data
        if len(price_data) < self.period:
            raise ValueError("Not enough data to calculate moving average")

        # Regular MA calculation
        sma_values = (
            np.convolve(price_data, np.ones(self.period), "valid") / self.period
        )
        return sma_values.tolist()


class MACD(Indicator):
    def __init__(
        self,
        short_period: int = 12,
        long_period: int = 26,
        signal_period: int = 9,
        use_returns: bool = False,
    ):
        super().__init__(short_period, use_returns)
        self.long_period = long_period
        self.signal_period = signal_period

    @property
    def name(self) -> IndicatorName:
        return IndicatorName(
            base_name="MACD",
            parameters=[self.period, self.long_period, self.signal_period],
            uses_returns=self.use_returns,
        )

    def compute(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        if len(data) < self.long_period:
            raise ValueError("Not enough data to calculate MACD")

        # Select the appropriate column based on use_returns
        if self.use_returns:
            price_data = np.array(data["close_returns"].values, dtype=float)
        else:
            price_data = np.array(data["close"].values, dtype=float)

        # Check if we have enough data
        if len(price_data) < self.long_period:
            raise ValueError("Not enough data to calculate MACD")

        ema_short = EMA(self.period, self.use_returns)._calculate_ema(price_data)
        ema_long = EMA(self.long_period, self.use_returns)._calculate_ema(price_data)

        # Ensure both EMAs are the same length
        min_length = min(len(ema_short), len(ema_long))
        ema_short = ema_short[-min_length:]
        ema_long = ema_long[-min_length:]

        macd_line = ema_short - ema_long

        signal_line = EMA(self.signal_period, self.use_returns)._calculate_ema(
            macd_line
        )

        # Ensure MACD and signal line are the same length
        min_length = min(len(macd_line), len(signal_line))
        macd_line = macd_line[-min_length:]
        signal_line = signal_line[-min_length:]

        macd_histogram = macd_line - signal_line

        return {
            "macd_line": macd_line.tolist(),
            "signal_line": signal_line.tolist(),
            "macd_histogram": macd_histogram.tolist(),
        }


class MFI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period + 1:
            raise ValueError("Not enough data to calculate MFI")

        # Select correct columns based on use_returns
        if self.use_returns:
            high = data["high_returns"].astype(float).values
            low = data["low_returns"].astype(float).values
            close = data["close_returns"].astype(float).values
            volume = (
                data["volume"].astype(float).values
            )  # Volume typically isn't transformed
        else:
            high = data["high"].astype(float).values
            low = data["low"].astype(float).values
            close = data["close"].astype(float).values
            volume = data["volume"].astype(float).values

        # Convert arrays to numpy arrays with explicit float dtype
        high_np = np.array(high, dtype=float)
        low_np = np.array(low, dtype=float)
        close_np = np.array(close, dtype=float)
        volume_np = np.array(volume, dtype=float)

        # Calculate typical price
        typical_price = (high_np + low_np + close_np) / 3.0
        raw_money_flow = typical_price * volume_np

        pos_money_flow = []
        neg_money_flow = []
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                pos_money_flow.append(raw_money_flow[i])
                neg_money_flow.append(0)
            elif typical_price[i] < typical_price[i - 1]:
                pos_money_flow.append(0)
                neg_money_flow.append(raw_money_flow[i])
            else:
                pos_money_flow.append(0)
                neg_money_flow.append(0)

        mfi = []
        for i in range(self.period - 1, len(pos_money_flow)):
            pos_flow = np.sum(pos_money_flow[i - self.period + 1 : i + 1])
            neg_flow = np.sum(neg_money_flow[i - self.period + 1 : i + 1])
            if neg_flow == 0:
                mfi_value = 100.0
            else:
                money_flow_ratio = pos_flow / neg_flow
                mfi_value = 100 - (100 / (1 + money_flow_ratio))
            mfi.append(float(mfi_value))

        return mfi


class OBV(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < 2:
            raise ValueError("Not enough data to calculate OBV")

        # Select correct column based on use_returns
        if self.use_returns:
            close = data["close_returns"].astype(float).values
        else:
            close = data["close"].astype(float).values

        volume = data["volume"].astype(float).values

        obv = [0.0]  # Start with a float instead of int
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(float(obv[-1] + volume[i]))
            elif close[i] < close[i - 1]:
                obv.append(float(obv[-1] - volume[i]))
            else:
                obv.append(float(obv[-1]))

        return obv


class RSI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate RSI")

        # Select correct column based on use_returns
        if self.use_returns:
            closes = np.array(data["close_returns"].values, dtype=float)
        else:
            closes = np.array(data["close"].values, dtype=float)

        deltas = np.diff(closes)
        seed = deltas[: self.period]
        up = seed[seed >= 0].sum() / self.period
        down = -seed[seed < 0].sum() / self.period
        rs = up / down if down != 0 else float("inf")
        rsi = np.zeros_like(closes)
        rsi[: self.period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(self.period, len(closes)):
            delta = deltas[i - 1]
            up = (up * (self.period - 1) + max(delta, 0)) / self.period
            down = (down * (self.period - 1) + max(-delta, 0)) / self.period
            rs = up / down if down != 0 else float("inf")
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        return rsi[self.period - 1 :].tolist()


class Stochastic(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        # This will compute %K line of the Stochastic Oscillator.
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate Stochastic Oscillator")

        # Select correct columns based on use_returns
        if self.use_returns:
            closes = data["close_returns"].astype(float).values
            highs = data["high_returns"].astype(float).values
            lows = data["low_returns"].astype(float).values
        else:
            closes = data["close"].astype(float).values
            highs = data["high"].astype(float).values
            lows = data["low"].astype(float).values

        stoch = []
        for i in range(self.period - 1, len(closes)):
            current_high = np.max(highs[i - self.period + 1 : i + 1])
            current_low = np.min(lows[i - self.period + 1 : i + 1])
            if current_high - current_low == 0:
                stoch.append(0.0)
            else:
                percent_k = (
                    (closes[i] - current_low) / (current_high - current_low) * 100
                )
                stoch.append(percent_k)
        return stoch
