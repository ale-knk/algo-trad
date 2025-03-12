from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class Indicator(ABC):
    def __init__(self, period: int):
        self.period = period

    @property
    def name(self) -> str:
        """
        Returns the name of the indicator with its parameters.
        Can be overridden in subclasses to provide more specific names.
        """
        return f"{self.__class__.__name__}_{self.period}"

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[List[float], Dict[str, List[float]]]:
        """
        Compute the indicator values based on the input DataFrame.

        Args:
            data: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Either a list of float values or a dictionary of lists
        """
        pass


class MA(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate moving average")

        closes = np.array(data["close"].values, dtype=float)
        sma_values = np.convolve(closes, np.ones(self.period), "valid") / self.period
        return sma_values.tolist()


class EMA(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate exponential moving average")

        closes = np.array(data["close"].values, dtype=float)
        ema_values = self._calculate_ema(closes)
        return ema_values.tolist()

    def _calculate_ema(self, data: np.ndarray) -> np.ndarray:
        ema = np.zeros_like(data)
        alpha = 2 / (self.period + 1)
        ema[: self.period] = np.mean(data[: self.period])
        for i in range(self.period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema  # Return full length array instead of slicing


class RSI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate RSI")

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


class MACD(Indicator):
    def __init__(
        self, short_period: int = 12, long_period: int = 26, signal_period: int = 9
    ):
        super().__init__(short_period)
        self.long_period = long_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return f"MACD_{self.period}_{self.long_period}_{self.signal_period}"

    def compute(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        if len(data) < self.long_period:
            raise ValueError("Not enough data to calculate MACD")

        closes = np.array(data["close"].values, dtype=float)

        ema_short = EMA(self.period)._calculate_ema(closes)
        ema_long = EMA(self.long_period)._calculate_ema(closes)

        # Ensure both EMAs are the same length
        min_length = min(len(ema_short), len(ema_long))
        ema_short = ema_short[-min_length:]
        ema_long = ema_long[-min_length:]

        macd_line = ema_short - ema_long

        signal_line = EMA(self.signal_period)._calculate_ema(macd_line)

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


class BollingerBands(Indicator):
    def __init__(self, period: int = 20, std_dev: float = 2):
        super().__init__(period)
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return f"BB_{self.period}_{self.std_dev}"

    def compute(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate Bollinger Bands.")

        closes = np.array(data["close"].values, dtype=float)

        sma = np.convolve(closes, np.ones(self.period) / self.period, mode="valid")
        std_dev = np.array(
            [np.std(closes[i : i + self.period]) for i in range(len(sma))]
        )
        upper_band = sma + self.std_dev * std_dev
        lower_band = sma - self.std_dev * std_dev

        return {
            "lower_band": lower_band.tolist(),
            "middle_band": sma.tolist(),
            "upper_band": upper_band.tolist(),
        }


class ATR(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period + 1:
            raise ValueError("Not enough data to calculate ATR")
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
        atr = [np.mean(tr[: self.period])]
        for i in range(self.period, len(tr)):
            atr.append((atr[-1] * (self.period - 1) + tr[i]) / self.period)
        return atr


class ADX(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period + 1:
            raise ValueError("Not enough data to calculate ADX")
        high = data["high"].astype(float).values
        low = data["low"].astype(float).values
        close = data["close"].astype(float).values

        tr = []
        plus_dm = []
        minus_dm = []
        for i in range(1, len(close)):
            current_tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr.append(current_tr)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0)
            minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0)

        # Wilder smoothing
        atr = [np.sum(tr[: self.period])]
        plus_di = []
        minus_di = []
        for i in range(self.period, len(tr)):
            atr_val = atr[-1] - (atr[-1] / self.period) + tr[i]
            atr.append(atr_val)
        atr = np.array(atr[self.period - 1 :])  # Align with subsequent calculations

        # Calculate smoothed plus and minus DM
        smooth_plus_dm = [np.sum(plus_dm[: self.period])]
        smooth_minus_dm = [np.sum(minus_dm[: self.period])]
        for i in range(self.period, len(plus_dm)):
            smooth_plus_dm.append(
                smooth_plus_dm[-1] - (smooth_plus_dm[-1] / self.period) + plus_dm[i]
            )
            smooth_minus_dm.append(
                smooth_minus_dm[-1] - (smooth_minus_dm[-1] / self.period) + minus_dm[i]
            )
        smooth_plus_dm = np.array(smooth_plus_dm)
        smooth_minus_dm = np.array(smooth_minus_dm)

        # Calculate DI+ and DI-
        di_plus = 100 * (smooth_plus_dm / atr)
        di_minus = 100 * (smooth_minus_dm / atr)
        dx = (
            100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        )  # avoid division by zero

        # ADX as Wilder smoothing of DX
        adx = [np.mean(dx[: self.period])]
        for i in range(self.period, len(dx)):
            adx.append((adx[-1] * (self.period - 1) + dx[i]) / self.period)
        return adx


class OBV(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < 2:
            raise ValueError("Not enough data to calculate OBV")
        close = data["close"].astype(float).values
        volume = data["volume"].astype(float).values

        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return obv


class MFI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period + 1:
            raise ValueError("Not enough data to calculate MFI")
        high = data["high"].astype(float).values
        low = data["low"].astype(float).values
        close = data["close"].astype(float).values
        volume = data["volume"].astype(float).values

        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

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
            mfi.append(mfi_value)
        return mfi


class Stochastic(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        # This will compute %K line of the Stochastic Oscillator.
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate Stochastic Oscillator")
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


class CCI(Indicator):
    def compute(self, data: pd.DataFrame) -> List[float]:
        if len(data) < self.period:
            raise ValueError("Not enough data to calculate CCI")
        high = data["high"].astype(float).values
        low = data["low"].astype(float).values
        close = data["close"].astype(float).values

        typical_price = (high + low + close) / 3.0
        cci = []
        for i in range(self.period - 1, len(typical_price)):
            tp_slice = typical_price[i - self.period + 1 : i + 1]
            sma = np.mean(tp_slice)
            mean_deviation = np.mean(np.abs(tp_slice - sma))
            if mean_deviation == 0:
                cci_value = 0.0
            else:
                cci_value = (typical_price[i] - sma) / (0.015 * mean_deviation)
            cci.append(cci_value)
        return cci
