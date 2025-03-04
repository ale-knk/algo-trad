import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Any
from pymongo import MongoClient
from pytrad.db import MongoDBHandler
from pytrad.indicator import MA, EMA, BollingerBands, RSI, MACD

class Candle:
    def __init__(self, time: datetime, open_price: float, high: float, low: float, close: float, volume: float):
        self.time = time   
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self):
        return (f"Candle(time={self.time}, open={self.open}, high={self.high}, "
                f"low={self.low}, close={self.close}, volume={self.volume})")

    def is_valid(self) -> bool:
        return self.low <= self.open <= self.high and self.low <= self.close <= self.high

    @classmethod
    def from_document(cls, doc: dict) -> "Candle":

        return cls(
            time=doc["time"],
            open_price=doc["open"],
            high=doc["high"],
            low=doc["low"],
            close=doc["close"],
            volume=doc["volume"]
        )

class CandleCollection:
    def __init__(self, candles: List[Candle] = None):
        self.candles: List[Candle] = candles if candles is not None else []

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CandleCollection(self.candles[index])
        elif isinstance(index, int):
            if index < 0:
                index += len(self.candles)
            if index < 0 or index >= len(self.candles):
                raise IndexError("Index out of range.")
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
        
    def __len__(self) -> int:
        return len(self.candles)
    
    def add_candle(self, candle: Candle):
        self.candles.append(candle)

    def remove_candle(self, index: int):
        if 0 <= index < len(self.candles):
            del self.candles[index]
        else:
            raise IndexError("Index out of range.")

    def get_last(self, n: int) -> List[Candle]:
        return self.candles[-n:] if n <= len(self.candles) else self.candles

    @classmethod        
    def from_db(cls,
                currency_pair: str = "EURUSD", 
                timeframe: str = "M15",
                start_date: datetime = datetime(2009, 2, 10), 
                end_date: datetime = datetime(2009, 2, 20)) -> "CandleCollection":
        
        mongodb_handler = MongoDBHandler(collection_name=f'{currency_pair}_{timeframe}')
        query = {}
        if start_date and end_date:
            query = {"time": {"$gte": start_date, "$lte": end_date}}
        
        cursor = mongodb_handler.collection.find(query).sort("time", 1)
        candles = []
        for doc in cursor:
            doc.pop("_id", None)
            candle = Candle.from_document(doc)
            candles.append(candle)
        return cls(candles)
    
    def plot(self, indicators=None):
        if not self.candles:
            raise ValueError("No hay candles para plotear.")

        data = {
            "Open": [candle.open for candle in self.candles],
            "High": [candle.high for candle in self.candles],
            "Low": [candle.low for candle in self.candles],
            "Close": [candle.close for candle in self.candles],
            "Volume": [candle.volume for candle in self.candles]
        }
        index = [candle.time for candle in self.candles]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        df.index.name = "Date"

        # Determine number of rows based on indicators
        overlay_indicators = []
        separate_indicators = []
        
        if indicators:
            for ind in indicators:
                if isinstance(ind, (MA, EMA)) or (hasattr(ind, 'plot_overlay') and ind.plot_overlay):
                    overlay_indicators.append(ind)
                else:
                    separate_indicators.append(ind)
        
        num_rows = 2 + len(separate_indicators)  # Candles + Volume + Separate indicators
        row_heights = [0.5] + [0.2] * (num_rows - 1)
        
        # Create subplots
        fig = make_subplots(
            rows=num_rows, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=row_heights
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candlestick"
            ),
            row=1, col=1
        )

        # Add overlay indicators on the same subplot as candlesticks
        if overlay_indicators:
            for ind in overlay_indicators:
                if isinstance(ind, MA):
                    ma_values = ind.compute(self)
                    # Adjust for the length difference
                    x_values = df.index[-(len(ma_values)):]
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=ma_values,
                            mode='lines',
                            name=f'MA({ind.period})',
                            line=dict(width=1.5)
                        ),
                        row=1, col=1
                    )
                elif isinstance(ind, EMA):
                    ema_values = ind.compute(self)
                    # Adjust for the length difference
                    x_values = df.index[-(len(ema_values)):]
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=ema_values,
                            mode='lines',
                            name=f'EMA({ind.period})',
                            line=dict(width=1.5, dash='dash')
                        ),
                        row=1, col=1
                    )
                elif isinstance(ind, BollingerBands):
                    bb_values = ind.compute(self)
                    # Adjust for the length difference
                    x_values = df.index[-(len(bb_values["middle_band"])):]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=bb_values["upper_band"],
                            mode='lines',
                            name=f'BB Upper({ind.period}, {ind.std_dev})',
                            line=dict(width=1, color='rgba(250, 128, 114, 0.7)')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=bb_values["middle_band"],
                            mode='lines',
                            name=f'BB Middle({ind.period})',
                            line=dict(width=1, color='rgba(128, 128, 128, 0.7)')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=bb_values["lower_band"],
                            mode='lines',
                            name=f'BB Lower({ind.period}, {ind.std_dev})',
                            line=dict(width=1, color='rgba(173, 216, 230, 0.7)')
                        ),
                        row=1, col=1
                    )

        # Add volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )

        # Add separate indicators
        current_row = 3
        for ind in separate_indicators:
            if isinstance(ind, RSI):
                rsi_values = ind.compute(self)
                # Adjust for the length difference
                x_values = df.index[-(len(rsi_values)):]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=rsi_values,
                        mode='lines',
                        name=f'RSI({ind.period})',
                        line=dict(color='purple')
                    ),
                    row=current_row, col=1
                )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="red"),
                    x0=x_values[0], y0=70, x1=x_values[-1], y1=70,
                    row=current_row, col=1
                )
                
                fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="green"),
                    x0=x_values[0], y0=30, x1=x_values[-1], y1=30,
                    row=current_row, col=1
                )
                
                # Update y-axis range
                fig.update_yaxes(range=[0, 100], row=current_row, col=1, title_text="RSI")
                
            elif isinstance(ind, MACD):
                macd_values = ind.compute(self)
                # Adjust for the length difference
                x_values = df.index[-(len(macd_values["macd_line"])):]
                
                # MACD Line
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=macd_values["macd_line"],
                        mode='lines',
                        name=f'MACD({ind.period},{ind.long_period})',
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                
                # Signal Line
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=macd_values["signal_line"],
                        mode='lines',
                        name=f'Signal({ind.signal_period})',
                        line=dict(color='red')
                    ),
                    row=current_row, col=1
                )
                
                # Histogram
                fig.add_trace(
                    go.Bar(
                        x=x_values,
                        y=macd_values["macd_histogram"],
                        name='Histogram',
                        marker=dict(
                            color=['rgba(0,255,0,0.7)' if val >= 0 else 'rgba(255,0,0,0.7)' for val in macd_values["macd_histogram"]]
                        )
                    ),
                    row=current_row, col=1
                )
                
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                
            current_row += 1

        # Update layout
        fig.update_layout(
            # title={
            #     'text': 'Price Chart with Indicators',
            #     'y': 0.99,  # Ajustado para dejar espacio a la leyenda
            #     'x': 0.5,
            #     'xanchor': 'center',
            #     'yanchor': 'top'
            # },
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=250 + 200 * (num_rows - 1), 
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80)  # Aumentar el margen superior para dar más espacio
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        fig.show()