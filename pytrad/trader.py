from datetime import datetime
from pytrad.strategy import Strategy

class Trade:
    def __init__(self, entry_price: float, vol: int, trade_type: str, sl_tp_ratio: float = None, stop_loss: float = None, take_profit: float = None, entry_time: datetime = None):
        self.entry_price = entry_price
        self.vol = vol
        self.trade_type = trade_type
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.is_open = True  
        self.exit_price = None
        self.entry_time = entry_time
        self.exit_time = None
        self.pnl = 0.0
        self.status = "OPENED"

        if sl_tp_ratio is not None:
            if trade_type == "LONG":
                self.stop_loss = self.entry_price - (self.entry_price * sl_tp_ratio)
                self.take_profit = self.entry_price + (self.entry_price * sl_tp_ratio)
            if trade_type == "SHORT":
                self.take_profit = self.entry_price - (self.entry_price * sl_tp_ratio)
                self.stop_loss = self.entry_price + (self.entry_price * sl_tp_ratio)

    def calculate_pnl(self, current_price: float) -> float:
        if self.trade_type == "LONG":
            return (current_price - self.entry_price) * self.vol
        elif self.trade_type == "SHORT":
            return (self.entry_price - current_price) * self.vol
        return 0.0

    def check_stop_loss_take_profit(self, current_price: float) -> str:
        if self.stop_loss and ((self.trade_type == "LONG" and current_price <= self.stop_loss) or
                               (self.trade_type == "SHORT" and current_price >= self.stop_loss)):
            return "STOP LOSS"
        if self.take_profit and ((self.trade_type == "LONG" and current_price >= self.take_profit) or
                                 (self.trade_type == "SHORT" and current_price <= self.take_profit)):
            return "TAKE PROFIT"
        return "NONE"

    def close(self, exit_price: float, status: str, exit_time: datetime = None):
        self.is_open = False
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = self.calculate_pnl(exit_price)
        self.status = status

    def get_trade_duration(self):
        return self.exit_time - self.entry_time
    
class Trader:
    def __init__(self, strategy: Strategy, initial_balance: float = 1000.0):
        self.strategy = strategy
        self.balance = initial_balance
        self.trade_history = []  
        self.open_trades = []

    def execute_trade(self, entry_price: float, trade_type: str, vol: int, sl_tp_ratio: float = None, stop_loss: float = None, take_profit: float = None, entry_time: datetime = None):
        if self.balance >= entry_price * vol:
            trade = Trade(entry_price=entry_price, vol=vol, trade_type=trade_type, sl_tp_ratio=sl_tp_ratio, stop_loss=stop_loss, take_profit=take_profit, entry_time=entry_time)
            self.open_trades.append(trade)

    def check_open_trades(self, current_price, current_time: datetime = None):
        for trade in self.open_trades[:]:
            result = trade.check_stop_loss_take_profit(current_price)
            if result == "STOP LOSS" or result == "TAKE PROFIT":
                trade.close(exit_price=current_price, status=result, exit_time=current_time)
                self.balance += trade.pnl
                self.trade_history.append(trade)
                self.open_trades.remove(trade)

