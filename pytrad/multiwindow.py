from datetime import datetime, timedelta
from typing import List, Optional

from pytrad.window import Window


class MultiWindow:
    def __init__(self, windows: List[Window], timeframes: List[str]):
        self.windows = windows
        self.timeframes = timeframes
        self._validate_timeframes()
        self.windows_kv = {tf: w for tf, w in zip(timeframes, windows)}

    @classmethod
    def from_dict(cls, data: dict):
        windows = list(data.values())
        timeframes = list(data.keys())
        return cls(windows, timeframes)

    @classmethod
    def from_db(
        cls, asset: str, timeframes: List[str], start_date: datetime, end_date: datetime
    ) -> "MultiWindow":
        """
        Create a MultiWindow by loading data from the database for multiple timeframes.

        Args:
            asset: Asset symbol (e.g., "EURUSD")
            timeframes: List of timeframes (e.g., ["M5", "M15", "H1"])
            start_date: Start date for the data
            end_date: End date for the data

        Returns:
            MultiWindow: An instance with windows for all specified timeframes
        """
        windows = []

        for timeframe in timeframes:
            window = Window.from_db(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            windows.append(window)

        return cls(windows=windows, timeframes=timeframes)

    def _validate_timeframes(self):
        if len(self.windows) != len(self.timeframes):
            raise ValueError("Number of windows and timeframes must be the same")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, key: str) -> Window:
        """
        Get a window by its timeframe key.

        Args:
            key: Timeframe key (e.g., "M15")

        Returns:
            Window: The window corresponding to the timeframe

        Raises:
            KeyError: If the timeframe is not found
        """
        if key not in self.windows_kv:
            raise KeyError(f"Timeframe '{key}' not found in MultiWindow")
        return self.windows_kv[key]

    def __setitem__(self, key: str, value: Window):
        """
        Set a window for a specific timeframe.

        Args:
            key: Timeframe key (e.g., "M15")
            value: Window instance
        """
        if key not in self.timeframes:
            # Add new timeframe
            self.timeframes.append(key)
            self.windows.append(value)
        else:
            # Update existing timeframe
            idx = self.timeframes.index(key)
            self.windows[idx] = value

        # Update the dictionary mapping
        self.windows_kv[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if a timeframe exists in this MultiWindow.

        Args:
            key: Timeframe key (e.g., "M15")

        Returns:
            bool: True if the timeframe exists, False otherwise
        """
        return key in self.windows_kv

    def get(self, key: str, default: Optional[Window] = None) -> Optional[Window]:
        """
        Get a window by its timeframe key with a default value.

        Args:
            key: Timeframe key (e.g., "M15")
            default: Default value to return if key is not found

        Returns:
            Window or default: The window corresponding to the timeframe or the default value
        """
        return self.windows_kv.get(key, default)

    def keys(self):
        """Return timeframes (keys) view"""
        return self.windows_kv.keys()

    def values(self):
        """Return windows (values) view"""
        return self.windows_kv.values()

    def items(self):
        """Return (timeframe, window) pairs view"""
        return self.windows_kv.items()

    def slice_generator(
        self,
        w_len: int,
        delta: timedelta,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        return_future_windows: bool = False,
        delta_future: timedelta = timedelta(days=1),
    ):
        start_dates = {
            tf: window.candles[w_len - 1].time for tf, window in self.windows_kv.items()
        }
        end_dates = {
            tf: window.candles[-1].time for tf, window in self.windows_kv.items()
        }

        latest_start = max(start_dates.values())
        earliest_end = min(end_dates.values())

        if return_future_windows:
            earliest_end -= delta_future

        if start_date and start_date < latest_start:
            raise ValueError(
                f"start_date cannot be earlier than the latest window start date ({latest_start})"
            )
        if end_date and end_date > earliest_end:
            if not return_future_windows:
                raise ValueError(
                    f"end_date cannot be later than the earliest window end date ({earliest_end})"
                )
            else:
                raise ValueError(
                    f"end_date cannot be later than the earliest window end date minus delta_future ({earliest_end})"
                )

        if not start_date:
            start_date = latest_start
        current_date = start_date

        if not end_date:
            end_date = earliest_end

        while True:
            if current_date > end_date:
                break
            windows = [
                self.windows_kv[tf].get_subwindow_by_datetime(current_date, w_len)
                for tf in self.timeframes
            ]
            windows = MultiWindow(windows, self.timeframes)

            if return_future_windows:
                future_windows = [
                    self.windows_kv[tf].get_subwindow_by_datetime(
                        current_date + delta_future, w_len
                    )
                    for tf in self.timeframes
                ]
                future_windows = MultiWindow(future_windows, self.timeframes)
                yield windows, future_windows
            else:
                yield windows

            current_date += delta
