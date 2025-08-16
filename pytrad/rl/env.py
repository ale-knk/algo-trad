import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from pytrad.multiwindow import MultiWindow
from pytrad.rl.sac import SACAgent
from pytrad.window import Window

logger = logging.getLogger(__name__)


class Action:
    def __init__(
        self,
        action_type: int,  # 0: Buy, 1: Sell, 2: Hold
        position_size: float = 1.0,
        stop_loss: float = 0.02,
        take_profit: float = 0.06,
    ):
        self.action_type = action_type
        self.position_size = self._validate_position_size(position_size)
        self.stop_loss = self._validate_stop_loss(stop_loss)
        self.take_profit = self._validate_take_profit(take_profit)

    def _validate_position_size(self, size):
        if not 0.0 <= size <= 1.0:
            size = max(0.0, min(size, 1.0))
        return size

    def _validate_stop_loss(self, sl):
        return max(0.001, sl)  # Mínimo 0.1%

    def _validate_take_profit(self, tp):
        return max(0.001, tp)  # Mínimo 0.1%

    def to_dict(self):
        return {
            "action_type": self.action_type,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }

    @classmethod
    def from_dict(cls, action_dict):
        return cls(
            action_type=action_dict["action_type"],
            position_size=action_dict.get("position_size", 1.0),
            stop_loss=action_dict.get("stop_loss", 0.02),
            take_profit=action_dict.get("take_profit", 0.06),
        )

    @classmethod
    def from_agent_output(cls, agent_output):
        """Convierte la salida del agente en una acción"""
        action_dict = {}
        # Extrae componentes del output del agente
        if isinstance(agent_output["action_type"], torch.Tensor):
            action_dict["action_type"] = (
                agent_output["action_type"].cpu().numpy().item()
            )
        else:
            action_dict["action_type"] = agent_output["action_type"]

        for key in ["position_size", "stop_loss", "take_profit"]:
            if key in agent_output:
                value = agent_output[key]
                if isinstance(value, torch.Tensor):
                    action_dict[key] = value.cpu().numpy().item()
                else:
                    action_dict[key] = value

        return cls.from_dict(action_dict)


class State:
    def __init__(
        self,
        features: np.ndarray,
        windows: Sequence[Window] = None,
        future_windows: Sequence[Window] = None,
        asset: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Clase que representa el estado del mercado.

        Args:
            features: Vector de características procesado por el feature extractor
            windows: Ventanas de datos originales (opcional, para referencia)
            asset: Activo asociado con este estado
            timestamp: Marca de tiempo del estado
        """
        self.features = features
        self.windows = windows
        self.future_windows = future_windows
        self.asset = asset
        self.timestamp = timestamp

    def to_tensor(self, device="cpu"):
        """Convierte el estado a un tensor"""
        return torch.FloatTensor(self.features).to(device)

    def to_numpy(self):
        """Devuelve las características como numpy array"""
        return self.features

    @classmethod
    def from_windows(cls, windows, feature_extractor, device="cpu"):
        """
        Crea un estado a partir de ventanas de datos y un extractor de características.

        Args:
            windows: Secuencia de objetos Window
            feature_extractor: TransformerFeatureExtractor
            device: Dispositivo para inferencia

        Returns:
            State: Nuevo objeto State
        """
        # Prepara los datos para el feature extractor
        data_batch = {}

        for window in windows:
            timeframe = window.timeframe

            # Extrae los datos OHLC y calcula retornos
            ohlc = np.array(
                [
                    [candle.open, candle.high, candle.low, candle.close]
                    for candle in window.candles
                ]
            )

            ohlc_ret = np.zeros_like(ohlc)
            ohlc_ret[1:] = (ohlc[1:] / ohlc[:-1]) - 1

            # Extrae indicadores si están disponibles
            if window.indicators:
                indicators = np.array(
                    [
                        [getattr(candle, ind_name) for ind_name in window.indicators]
                        for candle in window.candles
                    ]
                )
            else:
                indicators = np.zeros((len(window.candles), 0))

            # Extrae características de tiempo
            time_features = np.zeros((len(window.candles), 21))

            # Crea datos para este timeframe
            data_batch[timeframe] = {
                "ohlc_ret": torch.FloatTensor(ohlc_ret).unsqueeze(0).to(device),
                "indicators": torch.FloatTensor(indicators).unsqueeze(0).to(device),
                "time": torch.FloatTensor(time_features).unsqueeze(0).to(device),
                "asset": torch.LongTensor([[0]]).to(device),
                "market": torch.LongTensor([[0]]).to(device),
            }

        # Procesa a través del extractor de características
        with torch.no_grad():
            state_tensor = feature_extractor(data_batch)
            features = state_tensor.cpu().numpy()[0]

        # Crea y devuelve el objeto State
        return cls(
            features=features,
            windows=windows,
            asset=windows[0].asset if windows else None,
            timestamp=windows[0].candles[-1].time
            if windows and windows[0].candles
            else None,
        )


class TradingEnvironment:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        assets: list[str] = ["EURUSD"],
        timeframes: list[str] = ["M15", "M30", "H1", "H4"],
        window_size: int = 30,
        step_delta: timedelta = timedelta(hours=3),
        episode_delta: timedelta = timedelta(days=30),
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Trading environment for reinforcement learning with SAC.

        Args:
            dataset: MultiWindowDataset instance with market data
            feature_extractor: TransformerFeatureExtractor for state processing
            initial_balance: Starting capital
            commission_rate: Trading commission as percentage (e.g., 0.0001 for 0.01%)
            slippage: Slippage as percentage (e.g., 0.0001 for 0.01%)
            max_position_size: Maximum position size as a fraction of balance
            reward_scaling: Scaling factor for rewards
            window_size: Size of the observation window
            device: Device to run feature extraction on
        """

        self.assets = assets
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.step_delta = step_delta
        self.episode_delta = episode_delta

        self.current_asset = None
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.device = device

        self.windows_kv = {
            asset: MultiWindow.from_db(asset, timeframes, start_date, end_date)
            for asset in assets
        }

    def set_selected_asset(self):
        self.current_asset = np.random.choice(self.assets)

    def set_start_date(self):
        """
        Set a random start date between self.start_date and self.end_date.
        """
        if not self.current_asset:
            return None

        max_start_date = max(
            [
                window.candles[self.window_size - 1].time
                for window in self.windows_kv[self.current_asset].values()
            ]
        )
        min_end_date = min(
            [
                window.candles[-1].time
                for window in self.windows_kv[self.current_asset].values()
            ]
        )

        diff_seconds = (
            min_end_date - self.episode_delta - max_start_date
        ).total_seconds()
        while True:
            if diff_seconds > 0:
                potential_start_date = max_start_date + timedelta(
                    seconds=np.random.randint(0, int(diff_seconds))
                )
                if potential_start_date + self.episode_delta <= min_end_date:
                    self.current_timestamp = potential_start_date
                    break

    def reset(self) -> State:

        self.set_selected_asset()
        self.set_start_date()
        self.mw_gen = self.windows_kv[self.current_asset].slice_generator(
            w_len=self.window_size,
            delta=self.step_delta,
            start_date=self.current_timestamp,
            end_date=self.current_timestamp + self.episode_delta,
            return_future_windows=True,
        )

        self.current_windows, self.future_windows = next(self.mw_gen)

        self.current_state = State.from_windows(
            windows=self.current_windows,
            future_windows=self.future_windows,
            device=self.device,
        )

        
    def reset_new(self) -> State:
        """
        Reset the environment to a clean state with a random valid asset and date.

        Returns:
            State: Initial state observation
        """
        # Verificar que tenemos las fechas elegibles
        if not hasattr(self, "eligible_dates") or not self.eligible_dates:
            self.set_eligible_dates()

        # Si no hay fechas elegibles, lanzar excepción
        if not self.eligible_dates:
            raise ValueError("No hay fechas elegibles para ningún activo")

        # Escoger un activo al azar entre los que tienen fechas elegibles
        eligible_assets = list(self.eligible_dates.keys())
        self.current_asset = np.random.choice(eligible_assets)

        # Obtener rango de fechas válido para este activo
        date_range = self.eligible_dates[self.current_asset]
        start_date = date_range["start_date"]
        end_date = date_range["end_date"]

        # Calcular diferencia en segundos
        diff_seconds = (end_date - start_date).total_seconds()

        # Generar un punto aleatorio en ese rango (asegurándonos de dejar suficiente espacio para el window_size)
        # Restamos window_size días del final para asegurar que tenemos suficientes datos hacia adelante
        buffer_seconds = (
            self.window_size * 86400
        )  # window_size en días convertido a segundos
        valid_diff_seconds = max(0, diff_seconds - buffer_seconds)

        if valid_diff_seconds <= 0:
            # Si no hay suficiente espacio, usar la fecha de inicio
            random_seconds = 0
        else:
            # Generar un punto aleatorio en el rango válido
            random_seconds = np.random.randint(0, valid_diff_seconds)

        self.current_timestamp = start_date + timedelta(seconds=random_seconds)

        # Resetear variables del entorno
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trade_history = []
        self.portfolio_values = []
        self.rewards = []
        self.trades_executed = 0
        self.profitable_trades = 0

        # Obtener ventanas para todos los timeframes
        windows = []
        for timeframe in self.timeframes:
            window = Window.from_db(
                asset=self.current_asset,
                timeframe=timeframe,
                start_date=self.current_timestamp - timedelta(days=self.window_size),
                end_date=self.current_timestamp,
            )

            if len(window.candles) < self.window_size:
                logger.warning(
                    f"Window para {self.current_asset} en {timeframe} tiene menos de {self.window_size} velas"
                )

            windows.append(window)

        # Obtener el precio actual de la vela más reciente del timeframe más pequeño
        shortest_tf_idx = self.timeframes.index(
            min(self.timeframes, key=lambda x: int(x[1:]))
        )
        self.current_price = windows[shortest_tf_idx].candles[-1].close

        # Calcular valor inicial del portfolio
        self.portfolio_value = self.balance + self.position * self.current_price
        self.portfolio_values.append(self.portfolio_value)

        # Crear estado
        self.current_state = State.from_windows(
            windows=windows,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )
        self.done = False

        return self.current_state

    def reset(self, random_start: bool = False) -> State:
        """
        Reset the environment to a clean state.

        Args:
            random_start: If True, start at a random point in the dataset

        Returns:
            Initial state observation as a State object
        """

        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0

        # Reset tracking variables
        self.trade_history = []
        self.portfolio_values = []
        self.rewards = []
        self.trades_executed = 0
        self.profitable_trades = 0

        self._setup_windows_dict()

        # Set starting index
        if random_start:
            self.current_idx = np.random.randint(0, self.total_episodes)
        else:
            self.current_idx = 0

        # Get initial data
        asset, timestamp = self.dataset.global_mapping[self.current_idx]
        self.current_asset = asset
        self.current_timestamp = timestamp

        # Get initial windows and convert to state
        windows = self.dataset.get_windows(timestamp, asset)
        if windows is None:
            # Try next timestamp if this one fails
            self.current_idx += 1
            asset, timestamp = self.dataset.global_mapping[self.current_idx]
            self.current_asset = asset
            self.current_timestamp = timestamp
            windows = self.dataset.get_windows(timestamp, asset)

            if windows is None:
                raise ValueError("Could not get valid windows to start environment")

        # Get current price from the most recent candle of shortest timeframe
        shortest_tf_window = windows[
            -1
        ]  # Assumes timeframes are ordered from longest to shortest
        self.current_price = shortest_tf_window.candles[-1].close

        # Compute initial portfolio value
        self.portfolio_value = self.balance + self.position * self.current_price
        self.portfolio_values.append(self.portfolio_value)

        # Get state representation using State class
        self.current_state = State.from_windows(
            windows=windows,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )
        self.done = False

        return self.current_state

    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        Take an action in the environment.

        Args:
            action: Action object

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            logger.warning("Step called on a done environment. Call reset() first.")
            # Return a copy of the current state to avoid modifying it
            return (self.current_state, 0.0, True, {})

        # Record prior state
        prior_portfolio_value = self.portfolio_value

        # Process the action
        self._execute_action(action)

        # Move to next time step
        done, next_windows = self._advance_time()

        # If we have reached the end of the dataset
        if done:
            self.done = True
            # Calculate final portfolio value
            self.portfolio_value = self.balance + self.position * self.current_price
            # Close any open position with a market order
            if self.position != 0:
                self._close_position()

            # Final reward is the percentage change in portfolio value
            reward = (
                (self.portfolio_value / self.initial_balance) - 1.0
            ) * self.reward_scaling
            self.rewards.append(reward)

            info = self._get_episode_info()
            return self.current_state, reward, True, info

        # Create a new state from the advanced windows
        self.current_state = State.from_windows(
            windows=next_windows,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )

        # Calculate reward based on portfolio value change
        self.portfolio_value = self.balance + self.position * self.current_price
        self.portfolio_values.append(self.portfolio_value)

        # Basic reward based on portfolio change
        reward = (
            (self.portfolio_value / prior_portfolio_value) - 1.0
        ) * self.reward_scaling
        self.rewards.append(reward)

        # Get additional info
        info = {
            "timestamp": self.current_timestamp,
            "asset": self.current_asset,
            "price": self.current_price,
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "trades_executed": self.trades_executed,
        }

        return self.current_state, reward, self.done, info

    def _execute_action(self, action: Action):
        """
        Execute the trading action in the environment.

        Args:
            action: Action object with trade parameters
        """
        # Hold - do nothing
        if action.action_type == 2:
            return

        # Apply max position size constraint
        position_size = min(action.position_size, self.max_position_size)

        # Close existing position if changing direction
        if (self.position > 0 and action.action_type == 1) or (
            self.position < 0 and action.action_type == 0
        ):
            self._close_position()

        # Calculate position size in currency
        position_currency = self.balance * position_size

        # Calculate number of units to buy/sell
        units = position_currency / self.current_price

        # Apply slippage
        execution_price = self.current_price * (
            1 + self.slippage if action.action_type == 0 else 1 - self.slippage
        )

        # Apply commission
        commission = position_currency * self.commission_rate

        # Buy
        if (
            action.action_type == 0 and self.position <= 0
        ):  # Buy only if no long position
            # Update balance and position
            self.balance -= position_currency + commission
            self.position = units
            self.position_price = execution_price
            self.trades_executed += 1

            # Record trade
            self.trade_history.append(
                {
                    "timestamp": self.current_timestamp,
                    "action": "BUY",
                    "price": execution_price,
                    "units": units,
                    "commission": commission,
                    "balance": self.balance,
                    "stop_loss": self.position_price * (1 - action.stop_loss),
                    "take_profit": self.position_price * (1 + action.take_profit),
                }
            )

        # Sell
        elif (
            action.action_type == 1 and self.position >= 0
        ):  # Sell only if no short position
            # Update balance and position
            self.balance -= position_currency + commission
            self.position = -units
            self.position_price = execution_price
            self.trades_executed += 1

            # Record trade
            self.trade_history.append(
                {
                    "timestamp": self.current_timestamp,
                    "action": "SELL",
                    "price": execution_price,
                    "units": units,
                    "commission": commission,
                    "balance": self.balance,
                    "stop_loss": self.position_price * (1 + action.stop_loss),
                    "take_profit": self.position_price * (1 - action.take_profit),
                }
            )

    def _close_position(self):
        """Close the current position and update balance."""
        if self.position == 0:
            return

        # Calculate position value
        position_value = abs(self.position) * self.current_price

        # Apply slippage based on position direction
        if self.position > 0:  # Long position
            execution_price = self.current_price * (1 - self.slippage)
        else:  # Short position
            execution_price = self.current_price * (1 + self.slippage)

        # Calculate actual position value with slippage
        actual_position_value = abs(self.position) * execution_price

        # Calculate commission
        commission = actual_position_value * self.commission_rate

        # Calculate PnL
        if self.position > 0:  # Long position
            pnl = (
                actual_position_value
                - (self.position * self.position_price)
                - commission
            )
        else:  # Short position
            pnl = (
                (abs(self.position) * self.position_price)
                - actual_position_value
                - commission
            )

        # Update balance
        self.balance += actual_position_value - commission

        # Record trade
        action = "CLOSE_LONG" if self.position > 0 else "CLOSE_SHORT"
        self.trade_history.append(
            {
                "timestamp": self.current_timestamp,
                "action": action,
                "price": execution_price,
                "units": abs(self.position),
                "commission": commission,
                "pnl": pnl,
                "balance": self.balance,
            }
        )

        # Track profitable trades
        if pnl > 0:
            self.profitable_trades += 1

        # Reset position
        self.position = 0.0
        self.position_price = 0.0

    def _advance_time(self) -> Tuple[bool, Optional[Sequence[Window]]]:
        """
        Advance to the next time step.

        Returns:
            Tuple of (done, next_windows)
        """
        # Check for stop loss or take profit first
        self._check_sl_tp()

        # Increment the current index
        self.current_idx += 1

        # Check if we've reached the end of the dataset
        if self.current_idx >= self.total_episodes:
            return True, None

        # Get the next asset and timestamp
        asset, timestamp = self.dataset.global_mapping[self.current_idx]
        self.current_asset = asset
        self.current_timestamp = timestamp

        # Get the next windows
        windows = self.dataset.get_windows(timestamp, asset)

        # If windows is None, try to get the next valid windows
        attempts = 0
        while windows is None and attempts < 10:
            self.current_idx += 1
            if self.current_idx >= self.total_episodes:
                return True, None

            asset, timestamp = self.dataset.global_mapping[self.current_idx]
            self.current_asset = asset
            self.current_timestamp = timestamp
            windows = self.dataset.get_windows(timestamp, asset)
            attempts += 1

        # If we still couldn't get valid windows after attempts
        if windows is None:
            logger.error("Could not get valid windows after multiple attempts")
            return True, None

        # Update current price
        shortest_tf_window = windows[-1]
        self.current_price = shortest_tf_window.candles[-1].close

        return False, windows

    def _check_sl_tp(self):
        """Check and apply stop loss and take profit orders."""
        if self.position == 0:
            return

        # Get the last trade to find stop loss and take profit levels
        last_trade = next(
            (t for t in reversed(self.trade_history) if t["action"] in ["BUY", "SELL"]),
            None,
        )

        if not last_trade:
            return

        # Extract stop loss and take profit levels
        stop_loss = last_trade.get("stop_loss")
        take_profit = last_trade.get("take_profit")

        if stop_loss is None or take_profit is None:
            return

        # Check if price hit stop loss or take profit
        if self.position > 0:  # Long position
            if self.current_price <= stop_loss:
                logger.debug(f"Stop loss triggered at {self.current_price}")
                self._close_position()
            elif self.current_price >= take_profit:
                logger.debug(f"Take profit triggered at {self.current_price}")
                self._close_position()
        else:  # Short position
            if self.current_price >= stop_loss:
                logger.debug(f"Stop loss triggered at {self.current_price}")
                self._close_position()
            elif self.current_price <= take_profit:
                logger.debug(f"Take profit triggered at {self.current_price}")
                self._close_position()

    def _get_episode_info(self) -> Dict:
        """
        Get information about the completed episode.

        Returns:
            Dictionary with episode statistics
        """
        # Calculate returns
        final_return = (self.portfolio_value / self.initial_balance) - 1.0

        # Calculate Sharpe ratio (if we have enough data)
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            )
        else:
            sharpe_ratio = 0

        # Calculate win rate
        win_rate = (
            self.profitable_trades / self.trades_executed
            if self.trades_executed > 0
            else 0
        )

        return {
            "final_balance": self.balance,
            "final_portfolio_value": self.portfolio_value,
            "return": final_return,
            "trades_executed": self.trades_executed,
            "profitable_trades": self.profitable_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._calculate_max_drawdown(),
        }

    def _calculate_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown from peak to trough.

        Returns:
            Maximum drawdown as a percentage
        """
        if len(self.portfolio_values) <= 1:
            return 0.0

        # Convert to numpy array for easier calculations
        portfolio_values = np.array(self.portfolio_values)

        # Calculate the running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Calculate drawdown as (current - peak) / peak
        drawdowns = (portfolio_values - running_max) / running_max

        # Return the maximum drawdown (will be negative)
        return float(drawdowns.min())

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        In this implementation, just return the current state as text.

        Args:
            mode: Rendering mode

        Returns:
            String representation of the environment state
        """
        if mode != "human":
            return

        output = [
            f"Timestamp: {self.current_timestamp}",
            f"Asset: {self.current_asset}",
            f"Price: {self.current_price:.5f}",
            f"Balance: {self.balance:.2f}",
            f"Position: {self.position:.5f}",
            f"Position Price: {self.position_price:.5f}",
            f"Portfolio Value: {self.portfolio_value:.2f}",
            f"Trades Executed: {self.trades_executed}",
            f"Profitable Trades: {self.profitable_trades}",
        ]

        return "\n".join(output)

    def run_episode(self, agent: SACAgent, deterministic: bool = False) -> Dict:
        """
        Run a complete episode with the given agent.

        Args:
            agent: SACAgent instance to use for action selection
            deterministic: Whether to use deterministic actions

        Returns:
            Episode statistics
        """
        # Reset environment
        state = self.reset()
        done = False
        total_reward = 0

        while not done:
            # Select action
            state_tensor = state.to_tensor(device=self.device).unsqueeze(0)
            action_dict = agent.select_action(state_tensor, deterministic=deterministic)

            # Convert to Action object
            action = Action.from_agent_output(action_dict)

            # Take step in environment
            next_state, reward, done, info = self.step(action)

            # Update state
            state = next_state
            total_reward += reward

        episode_info = self._get_episode_info()
        episode_info["total_reward"] = total_reward

        return episode_info

    def set_eligible_dates(self):
        """
        Construye un diccionario que para cada activo (asset) almacena su fecha de inicio y fin válidas.

        El método calcula la intersección de fechas disponibles para todos los timeframes de cada activo,
        garantizando que cualquier fecha seleccionada dentro de ese rango tenga datos disponibles en todos
        los timeframes configurados.

        Returns:
            dict: Diccionario donde cada clave es un asset y cada valor es otro diccionario con 'start_date' y 'end_date'
        """
        self.eligible_dates = {}

        for asset in self.assets:
            # Inicializar con valores extremos (el rango más amplio posible)
            earliest_start = datetime.min
            latest_start = datetime.min
            earliest_end = datetime.max
            latest_end = datetime.max

            timeframe_ranges = {}

            # Obtener rangos de fechas para cada timeframe del activo
            for timeframe in self.timeframes:
                # Conseguir los datos para este timeframe y activo
                window = Window.from_db(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                if len(window.candles) == 0:
                    logger.warning(
                        f"No hay datos para {asset} en timeframe {timeframe}"
                    )
                    continue

                tf_start = window.candles[0].time
                tf_end = window.candles[-1].time

                timeframe_ranges[timeframe] = {"start": tf_start, "end": tf_end}

                # Actualizar la fecha de inicio más tardía (necesitamos que todos los timeframes tengan datos)
                if tf_start > latest_start:
                    latest_start = tf_start

                # Actualizar la fecha de fin más temprana
                if tf_end < earliest_end:
                    earliest_end = tf_end

            # Si no tenemos datos para ningún timeframe, saltamos este activo
            if not timeframe_ranges:
                logger.warning(
                    f"No hay datos disponibles para {asset} en ningún timeframe"
                )
                continue

            # Verificar que el rango sea válido (start debe ser anterior a end)
            if latest_start > earliest_end:
                logger.warning(
                    f"No hay un rango común de fechas para {asset} en todos los timeframes. "
                    f"Última fecha de inicio: {latest_start}, Primera fecha de fin: {earliest_end}"
                )
                continue

            # Guardar el rango común para este activo
            self.eligible_dates[asset] = {
                "start_date": latest_start,
                "end_date": earliest_end,
            }

            logger.info(
                f"Rango válido para {asset}: desde {latest_start} hasta {earliest_end}"
            )

        return self.eligible_dates
