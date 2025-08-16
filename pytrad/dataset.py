import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from pytrad.indexer import TimeframeIndexer
from pytrad.normalizer import FeatureNormalizer
from pytrad.window import Window

# Configure module logger
logger = logging.getLogger(__name__)


class SingleWindowDataset(Dataset):
    def __init__(
        self,
        window: Window,
        window_size: int = 30,
        stride: int = 1,
        indicators: Optional[List] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        max_samples_for_stats: int = 10000,
        future_timedelta: timedelta = timedelta(hours=24),
    ):
        logger.info(
            f"Initializing SingleAssetDataset for {window.asset}:{window.timeframe}"
        )
        self.window = window
        self.window_size = window_size
        self.stride = stride
        self.indicators = indicators or []
        self.normalizer = normalizer
        self._column_labels = None
        self.max_samples_for_stats = max_samples_for_stats
        self.start_date = window.start_date
        self.end_date = window.end_date
        self.future_timedelta = future_timedelta

        if self.indicators:
            # self.window.add_indicators(self.indicators)
            logger.debug(f"Computing {len(self.indicators)} indicators")
            self._compute_indicators()

        if self.normalizer:
            self.fit_normalizer()
            self.normalize()

        self._create_window_mapping()

        logger.info(
            f"SingleAssetDataset initialized with {len(self.window.candles)} candles"
        )

    @classmethod
    def from_db(
        cls,
        asset: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        window_size: int = 30,
        stride: int = 1,
        indicators: Optional[List] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        max_samples_for_stats: int = 10000,
        future_timedelta: timedelta = timedelta(hours=24),
    ):
        """
        Create a SingleAssetDataset by loading window data from the database.

        Args:
            asset: Asset symbol (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "M15")
            start_date: Start date for the data
            end_date: End date for the data
            window_size: Size of the sliding window
            stride: Step size for the sliding window
            indicators: List of indicators to compute
            normalizer: Optional pre-configured normalizer
            max_samples_for_stats: Maximum number of samples to use for normalization statistics
            future_timedelta: Time delta for future window (default 24 hours)

        Returns:
            SingleAssetDataset: Dataset initialized with window data from the database
        """
        logger.info(
            f"Creating SingleAssetDataset from db for {asset}:{timeframe} from {start_date} to {end_date}"
        )

        try:
            window = Window.from_db(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            logger.debug(
                f"Successfully loaded {len(window.candles)} candles from database"
            )
        except Exception as e:
            logger.error(f"Failed to load window data: {str(e)}")
            raise

        return cls(
            window=window,
            window_size=window_size,
            stride=stride,
            indicators=indicators,
            normalizer=normalizer,
            max_samples_for_stats=max_samples_for_stats,
            future_timedelta=future_timedelta,
        )

    @property
    def asset(self) -> str:
        if self.window.asset is None:
            return ""
        return self.window.asset

    @property
    def timeframe(self) -> str:
        if self.window.timeframe is None:
            return ""
        return self.window.timeframe

    @property
    def market(self) -> str:
        if self.window.market is None:
            return ""
        return self.window.market

    def _create_window_mapping(self) -> dict[int, datetime]:
        """
        Creates a mapping from window index to the end date of the window.
        Only includes windows for which a corresponding future window exists.

        Returns:
            dict: Mapping from window index to end datetime of the window
        """
        mapping = {}
        total_candles = len(self.window.candles)

        # Calculate the number of possible windows
        num_windows = (total_candles - self.window_size) // self.stride + 1

        index_counter = 0

        for i in range(num_windows):
            # Calculate the end index of the window
            end_idx = self.window_size - 1 + i * self.stride
            # Get the end date of the window
            end_date = self.window.candles[end_idx].time
            # Calculate the future date
            future_date = end_date + self.future_timedelta

            # Check if future window exists
            try:
                self.window.get_subwindow_by_datetime(future_date, self.window_size)
                # Map the window index to the end date if future window exists
                mapping[index_counter] = end_date
                index_counter += 1
            except ValueError:
                # Skip this window if future window doesn't exist
                continue

        self.mapping = mapping
        return mapping

    def _compute_indicators(self):
        try:
            self.window.add_indicators(self.indicators)
            initial_size = len(self.window.candles)
            self._clean_indicator_nans()
            final_size = len(self.window.candles)

            if initial_size != final_size:
                logger.info(
                    f"Removed {initial_size - final_size} rows with NaN values from indicators"
                )
        except Exception as e:
            logger.error(f"Error computing indicators: {str(e)}")
            raise

    def _clean_indicator_nans(self):
        df = self.window.to_df()

        # Find the first row where all values are valid (no NaNs)
        first_valid_idx = 0

        # For each row, check if there are any NaN values
        for idx, row in df.iterrows():
            if not row.isna().any():
                # Found the first row with no NaNs
                first_valid_idx = df.index.get_loc(idx)
                break

        if first_valid_idx > 0:
            logger.debug(f"Trimming {first_valid_idx} rows with NaN values")
            self.window = self.window[first_valid_idx:]

    def validate_normalizer(self):
        if self.normalizer is not None:
            if not isinstance(self.normalizer, FeatureNormalizer):
                raise ValueError(
                    "Normalizer must be an instance of FeatureNormalizer or None"
                )

    def fit_normalizer(self):
        """
        Fits the normalizer to the dataset without transforming the data.
        If the number of candles exceeds max_samples_for_stats, a random subset is used.
        """
        logger.debug("Fitting normalizer to dataset")
        try:
            if self.normalizer and not self.normalizer.is_fitted:
                df = self.window.to_df()

                # If we have more candles than max_samples_for_stats, take a random subset
                if len(df) > self.max_samples_for_stats:
                    logger.debug(
                        f"Taking random subset of {self.max_samples_for_stats} samples for normalization"
                    )
                    df = df.sample(n=self.max_samples_for_stats, random_state=42)

                self.normalizer.fit(df)
                logger.debug("Successfully fitted normalizer")

        except Exception as e:
            logger.error(f"Error during normalizer fitting: {str(e)}")
            raise

    def normalize(self):
        """
        Transforms the data using the fitted normalizer.
        """
        if self.normalizer:
            self.window.normalize(self.normalizer)

    def __getitem__(self, idx: int) -> Tuple[Window, Window]:
        """
        Returns a tuple of current window and future window based on the integer index.

        Args:
            idx: Integer index of the window

        Returns:
            Tuple[Window, Window]: A tuple containing (current_window, future_window)
        """
        # Get the end date from the window mapping
        if idx not in self.mapping:
            raise IndexError(f"Index {idx} is out of bounds")

        current_date = self.mapping[idx]
        future_date = current_date + self.future_timedelta

        current_window = self.window.get_subwindow_by_datetime(
            current_date, self.window_size
        )
        future_window = self.window.get_subwindow_by_datetime(
            future_date, self.window_size
        )

        return current_window, future_window

    def __len__(self) -> int:
        """
        Returns the number of available windows in the dataset.

        Returns:
            int: The number of windows in the dataset
        """
        return len(self.mapping)

    def save_normalizer(self, dir: str, metadata: Optional[dict] = None) -> str:
        """
        Save the normalizer to a file in the specified directory.

        Args:
            dir: Directory to save the normalizer
            metadata: Optional metadata to include in the saved file

        Returns:
            str: Path to the saved file
        """
        if self.normalizer is None:
            raise ValueError("No normalizer to save")

        if not os.path.exists(dir):
            os.makedirs(dir)

        file_path = os.path.join(dir, f"{self.asset}_{self.timeframe}.json")
        logger.debug(f"Saving normalizer to {file_path}")

        # Add basic metadata if not provided
        if metadata is None:
            metadata = {
                "asset": self.asset,
                "timeframe": self.timeframe,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            }

        self.normalizer.save(file_path, metadata=metadata)
        return file_path


class MultiWindowDataset(Dataset):
    def __init__(
        self,
        datasets: List[SingleWindowDataset],
        normalizer: FeatureNormalizer | None = None,
        future_timedelta: timedelta = timedelta(hours=24),
    ):
        logger.info(f"Initializing MultiAssetDataset with {len(datasets)} datasets")

        self.datasets = datasets
        self.normalizer = normalizer
        self.future_timedelta = future_timedelta
        self.datasets_kv = {}
        self.datasets_tuple = []
        self.window_size = datasets[0].window_size
        self.stride = datasets[0].stride

        # Group datasets by asset
        for dataset in datasets:
            asset = dataset.asset
            timeframe = dataset.timeframe

            self.datasets_tuple.append((asset, timeframe, dataset))

            if asset not in self.datasets_kv:
                self.datasets_kv[asset] = {}

            if timeframe not in self.datasets_kv[asset]:
                self.datasets_kv[asset][timeframe] = dataset
                logger.debug(f"Added dataset for {asset}:{timeframe}")

        self._validate_datasets()
        self.fit_normalizer()
        self.normalize()
        print("NORMALIZED!")
        self._create_global_mapping()
        logger.info(
            f"MultiAssetDataset initialized with {len(self.datasets_kv)} assets"
        )

    def fit_normalizer(self):
        df = pd.concat([dataset.window.to_df() for dataset in self.datasets])
        if self.normalizer:
            self.normalizer.fit(df)

    def normalize(self):
        for dataset in self.datasets:
            if self.normalizer:
                dataset.window.normalize(self.normalizer)

    def _validate_datasets(self):
        for _, _, dataset in self.datasets_tuple:
            if dataset.window_size != self.window_size:
                raise ValueError(
                    f"Window size mismatch: {dataset.window_size} != {self.window_size}"
                )
            if dataset.stride != self.stride:
                raise ValueError(f"Stride mismatch: {dataset.stride} != {self.stride}")
            if dataset.future_timedelta != self.future_timedelta:
                raise ValueError(
                    f"Future timedelta mismatch: {dataset.future_timedelta} != {self.future_timedelta}"
                )

    def _create_global_mapping(self):
        """
        Creates a global mapping from integer index to (asset, datetime) that ensures
        all timeframes have enough candles for each window across all assets.
        """
        self.global_mapping = {}
        self.timeframe_indexer = TimeframeIndexer()
        global_idx = 0

        # Create mapping for each asset
        for asset, timeframes in self.datasets_kv.items():
            # Find the shortest timeframe by comparing strings
            shortest_timeframe = min(
                timeframes.keys(), key=lambda tf: self.timeframe_indexer.encode(tf)
            )

            # Get the mapping from the shortest timeframe
            base_mapping = timeframes[shortest_timeframe].mapping

            # Filter datetimes that have enough candles in all timeframes
            for idx, dt in base_mapping.items():
                valid = True
                future_dt = dt + self.future_timedelta

                for timeframe, dataset in timeframes.items():
                    try:
                        # Check if we can get a full window for both current and future
                        dataset.window.get_subwindow_by_datetime(dt, self.window_size)
                        dataset.window.get_subwindow_by_datetime(
                            future_dt, self.window_size
                        )
                    except ValueError:
                        # Skip this datetime if any timeframe doesn't have enough candles
                        valid = False
                        break

                if valid:
                    # Store both asset and datetime in the global mapping
                    self.global_mapping[global_idx] = (asset, dt)
                    global_idx += 1

    def __getitem__(self, idx: int):
        """
        Returns a tuple of window tuples (one per timeframe) for the given integer index.
        Each inner tuple contains (current_window, future_window) for a specific timeframe.

        Args:
            idx: Integer index of the window

        Returns:
            tuple: A tuple of (current_window, future_window) tuples, one per timeframe
        """
        if idx not in self.global_mapping:
            raise IndexError(f"Index {idx} is out of bounds")

        asset, target_datetime = self.global_mapping[idx]
        return self.get_windows(target_datetime, asset)

    def get_windows(
        self, timestamp: datetime, asset: str
    ) -> Tuple[Tuple[Window, Window], ...]:
        """
        Gets windows for a specific asset and timestamp.

        Args:
            timestamp: Timestamp to get windows for
            asset: Asset to get windows for

        Returns:
            A tuple of (current_window, future_window) tuples, one for each timeframe
        """
        logger.debug(f"Getting windows for {asset} at {timestamp}")
        if asset not in self.datasets_kv:
            error_msg = f"Asset {asset} not found in datasets"
            logger.error(error_msg)
            raise ValueError(error_msg)

        out = []
        future_timestamp = timestamp + self.future_timedelta

        for timeframe, dataset in self.datasets_kv[asset].items():
            logger.debug(f"Retrieving window for {asset}:{timeframe}")
            try:
                current_window = dataset.window.get_subwindow_by_datetime(
                    timestamp, self.window_size
                )
                future_window = dataset.window.get_subwindow_by_datetime(
                    future_timestamp, self.window_size
                )

                if (
                    len(current_window) == self.window_size
                    and len(future_window) == self.window_size
                ):
                    out.append((current_window, future_window))
                else:
                    logger.warning(
                        f"Window for {asset}:{timeframe} at {timestamp} has incorrect size"
                    )
                    return None
            except ValueError as e:
                logger.warning(
                    f"Could not retrieve window for {asset}:{timeframe} at {timestamp}: {str(e)}"
                )
                return None

        return tuple(out)

    def __len__(self):
        """
        Returns the number of available windows across all assets and timeframes.
        """
        return len(self.global_mapping)

    @classmethod
    def from_db(
        cls,
        assets: list[str],
        timeframes: list[str],
        start_date: datetime,
        end_date: datetime,
        window_size: int = 30,
        stride: int = 1,
        indicators: Optional[List] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        max_samples_for_stats: int = 10000,
        future_timedelta: timedelta = timedelta(hours=24),
    ):
        logger.info(
            f"Creating MultiAssetDataset from db for {len(assets)} assets and {len(timeframes)} timeframes"
        )
        datasets = []
        for asset in assets:
            for timeframe in timeframes:
                logger.debug(f"Creating dataset for {asset}:{timeframe}")
                try:
                    datasets.append(
                        SingleWindowDataset.from_db(
                            asset,
                            timeframe,
                            start_date,
                            end_date,
                            window_size,
                            stride,
                            indicators,
                            max_samples_for_stats=max_samples_for_stats,
                            future_timedelta=future_timedelta,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create dataset for {asset}:{timeframe}: {str(e)}"
                    )
                    raise

        return cls(datasets, normalizer, future_timedelta)

    def save_normalizer(self, dir: str) -> str:
        """
        Save the single normalizer to the specified directory.

        Args:
            dir: Directory to save the normalizer

        Returns:
            str: Path to the saved file
        """
        if self.normalizer is None:
            raise ValueError("No normalizer to save")

        if not os.path.exists(dir):
            os.makedirs(dir)

        # Create a filename that represents all assets and timeframes
        assets = list(self.datasets_kv.keys())
        timeframes = set()
        for tf_dict in self.datasets_kv.values():
            timeframes.update(tf_dict.keys())

        filename = f"multi_{'_'.join(assets)}_{'_'.join(sorted(timeframes))}.json"
        file_path = os.path.join(dir, filename)

        logger.debug(f"Saving normalizer to {file_path}")

        # Add comprehensive metadata
        metadata = {
            "assets": assets,
            "timeframes": list(sorted(timeframes)),
            "window_size": self.window_size,
            "stride": self.stride,
            "start_date": min(
                [dataset.start_date for _, _, dataset in self.datasets_tuple]
            ).isoformat(),
            "end_date": max(
                [dataset.end_date for _, _, dataset in self.datasets_tuple]
            ).isoformat(),
        }

        self.normalizer.save(file_path, metadata=metadata)
        return file_path
