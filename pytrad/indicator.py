from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Indicator(ABC):
    def __init__(self, period: int):
        self.period = period

    @abstractmethod
    def compute(self, candle_collection: "CandleCollection") -> List[float]:
        pass

class MA(Indicator):
    def compute(self, candle_collection: "CandleCollection") -> List[float]:
        candles = candle_collection.candles
        if len(candles) < self.period:
            raise ValueError("No hay suficientes candles para calcular la media móvil")
        
        closes = np.array([candle.close for candle in candles])
        sma_values = np.convolve(closes, np.ones(self.period), 'valid') / self.period
        return sma_values.tolist()

class EMA(Indicator):
    def compute(self, candle_collection: "CandleCollection") -> List[float]:
        candles = candle_collection.candles
        if len(candles) < self.period:
            raise ValueError("Not enough candles to calculate exponential moving average")
        
        closes = np.array([candle.close for candle in candles])
        ema_values = self._calculate_ema(closes)
        return ema_values.tolist()

    def _calculate_ema(self, data: np.ndarray) -> np.ndarray:
        ema = np.zeros_like(data)
        alpha = 2 / (self.period + 1)
        ema[:self.period] = np.mean(data[:self.period])
        for i in range(self.period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema  # Return full length array instead of slicing

class RSI(Indicator):
    def compute(self, candle_collection: "CandleCollection") -> List[float]:
        candles = candle_collection.candles
        if len(candles) < self.period:
            raise ValueError("No hay suficientes candles para calcular el RSI")
        
        closes = np.array([candle.close for candle in candles])
        deltas = np.diff(closes)
        seed = deltas[:self.period]
        up = seed[seed >= 0].sum() / self.period
        down = -seed[seed < 0].sum() / self.period
        rs = up / down
        rsi = np.zeros_like(closes)
        rsi[:self.period] = 100. - 100. / (1. + rs)

        for i in range(self.period, len(closes)):
            delta = deltas[i - 1]
            up = (up * (self.period - 1) + max(delta, 0)) / self.period
            down = (down * (self.period - 1) + max(-delta, 0)) / self.period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi[self.period - 1:].tolist()

class MACD(Indicator):
    def __init__(self, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        super().__init__(short_period)
        self.long_period = long_period
        self.signal_period = signal_period
    
    def compute(self, candle_collection: "CandleCollection") -> dict:
        candles = candle_collection.candles
        if len(candles) < self.long_period:
            raise ValueError("Not enough candles to calculate MACD")
        
        closes = np.array([candle.close for candle in candles])
        
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
            "macd_histogram": macd_histogram.tolist()
        }

class BollingerBands(Indicator):
    def __init__(self, period: int = 20, std_dev: float = 2):
        super().__init__(period)
        self.std_dev = std_dev
    
    def compute(self, candle_collection: "CandleCollection") -> dict:
        candles = candle_collection.candles
        closes = np.array([candle.close for candle in candles])
        
        if len(closes) < self.period:
            raise ValueError("Not enough data to calculate Bollinger Bands.")
        
        sma = np.convolve(closes, np.ones(self.period) / self.period, mode='valid')
        std_dev = np.array([np.std(closes[i:i+self.period]) for i in range(len(sma))])
        upper_band = sma + self.std_dev * std_dev
        lower_band = sma - self.std_dev * std_dev

        return {
            "lower_band": lower_band.tolist(),
            "middle_band": sma.tolist(),
            "upper_band": upper_band.tolist()
        }
