from datetime import datetime
from tqdm import tqdm
from math import ceil
from typing import Iterator, Dict, Any, List
from joblib import Parallel, delayed

from math import ceil
from pytrad.candle import CandleCollection
from pytrad.trader import Trader, Trade
from pytrad.strategy import Strategy, MovingAverageCrossStrategy, RSIStrategy, MACDStrategy

class BackTestResults:
    def __init__(self, initial_balance: float, final_balance: float, trade_history: List[Trade]):
        self.initial_balance = initial_balance
        self.final_balance = final_balance
        self.trade_history = trade_history  

        self.total_trades = len(trade_history)
        self.profitable_trades = len([t for t in trade_history if t.pnl > 0])
        self.unprofitable_trades = len([t for t in trade_history if t.pnl <= 0])
        self.win_rate = (self.profitable_trades / self.total_trades) if self.total_trades > 0 else 0
        self.return_percentage = (final_balance - initial_balance) / initial_balance * 100

    def display(self):
        print("Resultados del Backtesting")
        print("--------------------------")
        print("Balance Inicial:", self.initial_balance)
        print("Balance Final:", self.final_balance)
        print("Total de Trades:", self.total_trades)
        print("Trades Ganadores:", self.profitable_trades)
        print("Trades Perdedoros:", self.unprofitable_trades)
        print("Tasa de Éxito:", f"{self.win_rate * 100:.2f}%")
        print("Retorno Total:", f"{self.return_percentage:.2f}%")
        

class BackTester:
    def __init__(self,
                 candle_collection: CandleCollection,
                 strategies: List[Strategy],
                 initial_balance: float = 1000.0,
                 vol_ratio: float = 0.1,
                 sl_tp_ratio: float = 0.05,
                 vote_threshold: float = 0.6):  # Umbral de consenso en porcentaje
        self.candle_collection = candle_collection
        self.strategies = strategies
        self.initial_balance = initial_balance
        self.vol_ratio = vol_ratio
        self.sl_tp_ratio = sl_tp_ratio
        self.vote_threshold = vote_threshold
        self.trader = Trader(strategy=None, initial_balance=self.initial_balance)

    def set_start_index(self):
        if len(self.strategies) == 1:
            self.start_index = self.strategies[0].start_index
        else:
            self.start_index = max(strategie.start_index for strategie in self.strategies)
        
    def set_required_votes(self):
        self.required_votes = ceil(len(self.strategies) * self.vote_threshold)
    
    def run(self) -> BackTestResults:
        self.set_start_index()
        self.set_required_votes()
        
        for i in tqdm(range(self.start_index, len(self.candle_collection)), desc="Progreso del Backtesting"):
            current_candles = self.candle_collection[:i+1]
            signals = [strategy.generate_signal(current_candles) for strategy in self.strategies]
            
            signal_counts = {
                "LONG": signals.count("LONG"),
                "SHORT": signals.count("SHORT"),
                "HOLD": signals.count("HOLD")
            }
            
            ensemble_signal = "HOLD"
            for signal, count in signal_counts.items():
                if count >= self.required_votes:
                    ensemble_signal = signal
                    break
            
            if ensemble_signal == "LONG" or ensemble_signal == "SHORT":
                vol = self.trader.balance * self.vol_ratio
                self.trader.execute_trade(
                    entry_price=current_candles[-1].close,
                    trade_type=ensemble_signal,
                    vol=vol,
                    sl_tp_ratio=self.sl_tp_ratio,
                    entry_time=current_candles[-1].time
                )
            
            self.trader.check_open_trades(
                current_price=current_candles[-1].close,
                current_time=current_candles[-1].time
            )

        results = BackTestResults(
            initial_balance=self.initial_balance,
            final_balance=self.trader.balance,
            trade_history=self.trader.trade_history
        )
        return results

def run_single_combination(candle_collection: "CandleCollection", param_combo: Dict[str, Any]) -> Dict[str, Any]:

    strategies = []
    for strat_conf in param_combo.get('strategies', []):
        strat_class = strat_conf['class']
        strat_kwargs = strat_conf.get('kwargs', {})
        strategy_instance = strat_class(**strat_kwargs)
        strategies.append(strategy_instance)

    vol_ratio = param_combo.get('vol_ratio', 0.1)
    sl_tp_ratio = param_combo.get('sl_tp_ratio', 0.05)
    vote_threshold = param_combo.get('vote_threshold', 0.6)
    initial_balance = param_combo.get('initial_balance', 1000.0)

    backtester = BackTester(
        candle_collection=candle_collection,
        strategies=strategies,
        initial_balance=initial_balance,
        vol_ratio=vol_ratio,
        sl_tp_ratio=sl_tp_ratio,
        vote_threshold=vote_threshold
    )

    # 4. Ejecutar el backtest
    results = backtester.run()

    # 5. Retornar los resultados junto con la combinación de parámetros
    return {
        'params': param_combo,
        'results': results
    }

class BackTesterHyperSearch:
    def __init__(self,
                 candle_collection: CandleCollection,
                 param_iterator: Iterator[Dict[str, Any]]):
        self.candle_collection = candle_collection
        self.param_iterator = param_iterator
        self.all_results = []  # Aquí guardaremos (param_combo, BackTestResults)

    def run_search(self, n_jobs: int = 4) -> List[Dict[str, Any]]:
        param_list = list(self.param_iterator)

        self.all_results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_combination)(self.candle_collection, combo)
            for combo in param_list
        )

        return self.all_results



if __name__ == "__main__":
    candle_collection = CandleCollection.from_db(
        currency_pair="EURUSD",
        timeframe="M15",
        start_date=datetime(2023, 2, 10),
        end_date=datetime(2023, 6, 20)
    )
    print("Total de velas cargadas:", len(candle_collection))

    def param_combinations():
        strategies_kwargs = [
            {
                'class': MovingAverageCrossStrategy,
                'kwargs': {
                    'short_period': [5,10],
                    'long_period': [20,30]
                }
            },
            {
                'class': RSIStrategy,
                'kwargs': {
                    'period': [10,20],
                }
            },
            {
                'class': MACDStrategy,
                'kwargs': {
                    'short_period': [5,10],
                    'long_period': [20,30]
                }
            }
        ]

        for vol_ratio in [0.05]:
            for sl_tp_ratio in [0.02]:
                for vote_threshold in [0.6]:
                    strategies = []
                    for strategy_class, kwargs in strategies_kwargs:
                        strategies.append(strategy_class(**kwargs))
                        
                        yield {
                            "strategies": strategies,
                            "vol_ratio": vol_ratio,
                            "sl_tp_ratio": sl_tp_ratio,
                            "vote_threshold": vote_threshold,
                        }

    hyper_search = BackTesterHyperSearch(
        candle_collection=candle_collection,
        param_iterator=param_combinations()
    )

    results_list = hyper_search.run_search(n_jobs=4)

    for result_item in results_list:
        combo = result_item['params']
        backtest_res = result_item['results']
        print("\n====================================")
        print("Parámetros probados:", combo)
        backtest_res.display()  