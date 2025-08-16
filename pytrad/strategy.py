from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from pytrad.indicator import Indicator, MA, EMA, RSI, MACD, BollingerBands
from typing import List

class Strategy(ABC):
    def __init__(self, name: str, indicators: List["Indicator"] = None, start_index: int = 20):
        if not name:
            raise ValueError("El nombre de la estrategia no puede estar vacío.")
        self.name = name
        self.indicators = indicators if indicators is not None else []
        self.last_signal = None
        self.start_index = start_index

    @abstractmethod
    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        pass

    def update_indicators(self, candle_collection: "CandleCollection"):
        computed_values = {}
        for indicator in self.indicators:
            computed_values[indicator.__class__.__name__] = indicator.compute(candle_collection)
        return computed_values
    
class MACrossStrategy(Strategy):
    def __init__(self, short_period: int, long_period: int, start_index: int = 20):
        if short_period >= long_period:
            raise ValueError("El período corto debe ser menor que el período largo.")
        indicators = [
            EMA(short_period),
            MA(long_period)
        ]
        super().__init__(name="MACrossStrategy", indicators=indicators, start_index=start_index)
        self.short_period = short_period
        self.long_period = long_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        ema_values = computed_values['ExponentialMovingAverage']
        sma_values = computed_values['MovingAverage']

        if len(ema_values) < 2 or len(sma_values) < 2:
            return "HOLD"

        if ema_values[-2] <= sma_values[-2] and ema_values[-1] > sma_values[-1]:
            return "LONG"
        elif ema_values[-2] >= sma_values[-2] and ema_values[-1] < sma_values[-1]:
            return "SHORT"
        else:
            return "HOLD"
        
class RSIStrategy(Strategy):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, start_index: int = 20):
        indicators = [RSI(period)]
        super().__init__(name="RSIStrategy", indicators=indicators, start_index=start_index)
        self.overbought = overbought
        self.oversold = oversold

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        rsi_values = computed_values['RSI']

        if len(rsi_values) < 2:
            return "HOLD"

        if rsi_values[-1] < self.oversold and rsi_values[-2] >= self.oversold:
            return "LONG"
        elif rsi_values[-1] > self.overbought and rsi_values[-2] <= self.overbought:
            return "SHORT"
        else:
            return "HOLD"

class MACD_Cross_Strategy(Strategy):
    def __init__(self, macd_params=(12, 26, 9), start_index: int = 26):
        indicators = [MACD(*macd_params)]
        super().__init__(name="MACD_Cross_Strategy", indicators=indicators, start_index=start_index)
        self.macd_params = macd_params

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        macd_data = computed_values['MACD']
        
        macd_line = macd_data["macd_line"]
        signal_line = macd_data["signal_line"]

        if len(macd_line) < 2 or len(signal_line) < 2:
            return "HOLD"

        macd_prev, macd_curr = macd_line[-2], macd_line[-1]
        signal_prev, signal_curr = signal_line[-2], signal_line[-1]

        if macd_prev <= signal_prev and macd_curr > signal_curr:
            return "LONG"
        elif macd_prev >= signal_prev and macd_curr < signal_curr:
            return "SHORT"
        else:
            return "HOLD"
        
class MACD_RSI_Strategy(Strategy):
    def __init__(self, macd_params=(12, 26, 9), rsi_period=14, start_index: int = 26):
        indicators = [
            MACD(*macd_params),
            RSI(rsi_period)
        ]
        super().__init__(name="MACD_RSI_Strategy", indicators=indicators, start_index=start_index)
        self.macd_params = macd_params
        self.rsi_period = rsi_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        macd_data = computed_values['MACD']
        rsi_values = computed_values['RelativeStrengthIndex']

        macd_histogram = macd_data["macd_histogram"]

        if len(macd_histogram) < 2 or len(rsi_values) < 2:
            return "HOLD"

        macd_prev, macd_curr = macd_histogram[-2], macd_histogram[-1]
        rsi_prev, rsi_curr = rsi_values[-2], rsi_values[-1]

        if macd_prev <= 0 and macd_curr > 0 and rsi_curr < 30:
            return "LONG"
        elif macd_prev >= 0 and macd_curr < 0 and rsi_curr > 70:
            return "SHORT"
        else:
            return "HOLD"


class MeanReversionRSIStrategy(Strategy):
    def __init__(self, rsi_period: int = 3, oversold_level: int = 20, exit_level: int = 60, start_index: int = 20):
        indicators = [RSI(rsi_period)]
        super().__init__(name="MeanReversionRSIStrategy", indicators=indicators, start_index=start_index)
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.exit_level = exit_level

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        rsi_values = computed_values['RelativeStrengthIndex']

        if len(rsi_values) < 2:
            return "HOLD"

        if rsi_values[-1] < self.oversold_level:
            return "LONG"
        elif rsi_values[-1] > self.exit_level:
            return "SHORT"
        else:
            return "HOLD"

class MACDMACrossoverStrategy(Strategy):
    def __init__(self, short_ma_period: int = 50, long_ma_period: int = 200, start_index: int = 200):
        indicators = [
            MA(short_ma_period),
            MA(long_ma_period),
            MACD()
        ]
        super().__init__(name="MACDMovingAverageCrossoverStrategy", indicators=indicators, start_index=start_index)
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        short_ma_values = computed_values['MovingAverage']
        long_ma_values = computed_values['MovingAverage']
        macd_values = computed_values['MACD']

        if len(short_ma_values) < 2 or len(long_ma_values) < 2 or len(macd_values) < 2:
            return "hold"

        # Verificar cruce de medias móviles
        golden_cross = short_ma_values[-2] <= long_ma_values[-2] and short_ma_values[-1] > long_ma_values[-1]
        death_cross = short_ma_values[-2] >= long_ma_values[-2] and short_ma_values[-1] < long_ma_values[-1]

        # Confirmación del MACD
        macd_trending_up = macd_values[-1] > macd_values[-2]
        macd_trending_down = macd_values[-1] < macd_values[-2]

        if golden_cross and macd_trending_up:
            return "buy"
        elif death_cross and macd_trending_down:
            return "sell"
        else:
            return "hold"


class TripleMovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_ma_period: int = 10, mid_ma_period: int = 50, long_ma_period: int = 200, start_index: int = 200):
        indicators = [
            MA(short_ma_period),
            MA(mid_ma_period),
            MA(long_ma_period)
        ]
        super().__init__(name="TripleMovingAverageCrossoverStrategy", indicators=indicators, start_index=start_index)
        self.short_ma_period = short_ma_period
        self.mid_ma_period = mid_ma_period
        self.long_ma_period = long_ma_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        short_ma_values = computed_values['MovingAverage']
        mid_ma_values = computed_values['MovingAverage']
        long_ma_values = computed_values['MovingAverage']

        if len(short_ma_values) < 1 or len(mid_ma_values) < 1 or len(long_ma_values) < 1:
            return "HOLD"

        if short_ma_values[-1] > mid_ma_values[-1] and mid_ma_values[-1] > long_ma_values[-1]:
            return "LONG"
        elif short_ma_values[-1] < mid_ma_values[-1] and mid_ma_values[-1] < long_ma_values[-1]:
            return "SHORT"
        else:
            return "HOLD"


class DivergenceMACDRSIStrategy(Strategy):
    def __init__(self, rsi_period: int = 14, start_index: int = 50):
        indicators = [
            MACD(),
            RSI(rsi_period)
        ]
        super().__init__(name="DivergenceMACDRSIStrategy", indicators=indicators, start_index=start_index)
        self.rsi_period = rsi_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        macd_values = computed_values['MACD']
        rsi_values = computed_values['RelativeStrengthIndex']
        closes = [candle.close for candle in candle_collection.candles]

        if len(macd_values) < 3 or len(rsi_values) < 3 or len(closes) < 3:
            return "HOLD"

        # Divergencia Alcista: Precio hace mínimos más bajos, pero MACD o RSI hacen mínimos más altos
        if closes[-1] < closes[-2] < closes[-3] and (macd_values[-1] > macd_values[-2] > macd_values[-3] or rsi_values[-1] > rsi_values[-2] > rsi_values[-3]):
            return "LONG"

        # Divergencia Bajista: Precio hace máximos más altos, pero MACD o RSI hacen máximos más bajos
        if closes[-1] > closes[-2] > closes[-3] and (macd_values[-1] < macd_values[-2] < macd_values[-3] or rsi_values[-1] < rsi_values[-2] < rsi_values[-3]):
            return "SHORT"

        return "HOLD"


class BollingerMeanReversionStrategy(Strategy):
    def __init__(self, bollinger_period: int = 20, rsi_period: int = 14, std_dev: float = 2, start_index: int = 20):
        indicators = [
            BollingerBands(bollinger_period, std_dev),
            RSI(rsi_period)
        ]
        super().__init__(name="BollingerMeanReversionStrategy", indicators=indicators, start_index=start_index)
        self.bollinger_period = bollinger_period
        self.rsi_period = rsi_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        bollinger_values = computed_values['BollingerBands']
        rsi_values = computed_values['RelativeStrengthIndex']
        closes = [candle.close for candle in candle_collection.candles]

        if len(bollinger_values) < 1 or len(rsi_values) < 1:
            return "HOLD"

        lower_band, _, upper_band = bollinger_values[-1]

        # Condiciones de compra y venta
        if closes[-1] < lower_band and rsi_values[-1] < 30:
            return "LONG"
        elif closes[-1] > upper_band and rsi_values[-1] > 70:
            return "SHORT"

        return "HOLD"


class BollingerBreakoutStrategy(Strategy):
    def __init__(self, bollinger_period: int = 20, std_dev: float = 2, start_index: int = 20):
        indicators = [
            BollingerBands(bollinger_period, std_dev),
            MACD()
        ]
        super().__init__(name="BollingerBreakoutStrategy", indicators=indicators, start_index=start_index)
        self.bollinger_period = bollinger_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        bollinger_values = computed_values['BollingerBands']
        macd_values = computed_values['MACD']
        closes = [candle.close for candle in candle_collection.candles]

        if len(bollinger_values) < 1 or len(macd_values) < 2:
            return "HOLD"

        lower_band, _, upper_band = bollinger_values[-1]
        macd_current = macd_values[-1]
        macd_previous = macd_values[-2]

        # Condiciones de compra y venta con confirmación del MACD
        if closes[-1] > upper_band and macd_current > macd_previous:
            return "LONG"
        elif closes[-1] < lower_band and macd_current < macd_previous:
            return "SHORT"

        return "HOLD"
    


class BollingerSqueezeStrategy(Strategy):
    def __init__(self, bollinger_period: int = 20, std_dev: float = 2, atr_period: int = 14, start_index: int = 20):
        indicators = [
            BollingerBands(bollinger_period, std_dev),
            AverageTrueRange(atr_period)
        ]
        super().__init__(name="BollingerSqueezeStrategy", indicators=indicators, start_index=start_index)
        self.bollinger_period = bollinger_period
        self.atr_period = atr_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        bollinger_values = computed_values['BollingerBands']
        atr_values = computed_values['AverageTrueRange']

        if len(bollinger_values) < 2 or len(atr_values) < 2:
            return "HOLD"

        lower_band_prev, _, upper_band_prev = bollinger_values[-2]
        lower_band, _, upper_band = bollinger_values[-1]

        atr_current = atr_values[-1]

        # Condición de estrechamiento de Bollinger Bands (squeeze)
        squeeze = (upper_band - lower_band) < (upper_band_prev - lower_band_prev)

        # Condiciones de compra y venta
        if squeeze and atr_current > atr_values[-2]:
            return "LONG"
        elif squeeze and atr_current < atr_values[-2]:
            return "SHORT"

        return "HOLD"

    
class BollingerMovingAverageCrossoverStrategy(Strategy):
    def __init__(self, bollinger_period: int = 20, ma_period: int = 20, std_dev: float = 2, start_index: int = 20):
        indicators = [
            BollingerBands(bollinger_period, std_dev),
            MA(ma_period)
        ]
        super().__init__(name="BollingerMovingAverageCrossoverStrategy", indicators=indicators, start_index=start_index)
        self.bollinger_period = bollinger_period
        self.ma_period = ma_period

    def generate_signal(self, candle_collection: "CandleCollection") -> str:
        computed_values = self.update_indicators(candle_collection)
        bollinger_values = computed_values['BollingerBands']
        ma_values = computed_values['MovingAverage']
        closes = [candle.close for candle in candle_collection.candles]

        if len(bollinger_values) < 1 or len(ma_values) < 1:
            return "HOLD"

        _, mid_band, _ = bollinger_values[-1]
        ma_current = ma_values[-1]

        # Condiciones de compra y venta
        if closes[-1] > mid_band and closes[-1] > ma_current:
            return "LONG"
        elif closes[-1] < mid_band and closes[-1] < ma_current:
            return "SHORT"

        return "HOLD"
    