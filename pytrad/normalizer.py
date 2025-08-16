import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureStats(ABC):
    """Abstract base class for feature normalization statistics."""

    @abstractmethod
    def fit(self, values: pd.Series) -> None:
        """Compute statistics from provided data values."""
        pass

    @abstractmethod
    def transform(self, values: pd.Series) -> pd.Series:
        """Transform data values using computed statistics."""
        pass

    def range_normalize(
        self,
        values: pd.Series,
        target_min: float,
        target_max: float,
        observed_min: float = None,
        observed_max: float = None,
    ) -> pd.Series:
        """Normalize values to a specific range based on observed min/max.

        Args:
            values: Series to normalize
            target_min: Target minimum value (e.g., 0 or -1)
            target_max: Target maximum value (e.g., 1)
            observed_min: Override for minimum observed value (if None, uses values.min())
            observed_max: Override for maximum observed value (if None, uses values.max())

        Returns:
            Series normalized to the target range
        """
        min_val = observed_min if observed_min is not None else values.min()
        max_val = observed_max if observed_max is not None else values.max()

        # Handle cases where min == max (constant values)
        if min_val == max_val:
            return pd.Series(
                np.full(len(values), (target_min + target_max) / 2), index=values.index
            )

        # Scale to target range
        normalized = (values - min_val) / (max_val - min_val)
        return normalized * (target_max - target_min) + target_min

    @abstractmethod
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary for serialization."""
        pass

    @classmethod
    @abstractmethod
    def from_stats_dict(cls, stats_dict: Dict[str, Any]) -> "FeatureStats":
        """Create an instance from a statistics dictionary."""
        pass


class ZScoreStats(FeatureStats):
    """Z-score normalization statistics."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        range_min: float = -1.0,
        range_max: float = 1.0,
    ):
        self.mean = mean
        self.std = std
        self.range_min = range_min
        self.range_max = range_max
        self.observed_min = None
        self.observed_max = None

    def fit(self, values: pd.Series) -> None:
        self.mean = float(values.mean())
        self.std = float(values.std()) or 1.0  # Avoid division by zero

        # Calculate z-scores for fitting range
        z_scores = (values - self.mean) / self.std
        self.observed_min = float(z_scores.min())
        self.observed_max = float(z_scores.max())

    def transform(self, values: pd.Series) -> pd.Series:
        # First apply z-score normalization
        z_scores = (values - self.mean) / self.std

        # Then normalize to target range
        return self.range_normalize(
            z_scores,
            self.range_min,
            self.range_max,
            self.observed_min,
            self.observed_max,
        )

    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }

    @classmethod
    def from_stats_dict(cls, stats_dict: Dict[str, Any]) -> "ZScoreStats":
        instance = cls(
            mean=stats_dict["mean"],
            std=stats_dict["std"],
            range_min=stats_dict.get("range_min", -1.0),
            range_max=stats_dict.get("range_max", 1.0),
        )
        instance.observed_min = stats_dict.get("observed_min")
        instance.observed_max = stats_dict.get("observed_max")
        return instance


class LogTransformStats(FeatureStats):
    """Log transform followed by Z-score normalization."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ):
        self.mean = mean
        self.std = std
        self.range_min = range_min
        self.range_max = range_max
        self.observed_min = None
        self.observed_max = None

    def fit(self, values: pd.Series) -> None:
        positive_mask = values > 0
        if positive_mask.any():
            log_values = np.log(values[positive_mask])
            self.mean = float(log_values.mean())
            self.std = float(log_values.std()) or 1.0

            # Apply log transform + z-score to calculate observed range
            result = values.copy()
            result.loc[positive_mask] = np.log(result.loc[positive_mask])
            normalized = (result - self.mean) / self.std
            self.observed_min = float(normalized.min())
            self.observed_max = float(normalized.max())
        else:
            # Fallback if no positive values
            self.mean = 0.0
            self.std = 1.0
            self.observed_min = 0.0
            self.observed_max = 1.0

    def transform(self, values: pd.Series) -> pd.Series:
        result = values.copy()
        positive_mask = result > 0
        if positive_mask.any():
            result.loc[positive_mask] = np.log(result.loc[positive_mask])

        # First apply z-score
        z_normalized = (result - self.mean) / self.std

        # Then normalize to target range
        return self.range_normalize(
            z_normalized,
            self.range_min,
            self.range_max,
            self.observed_min,
            self.observed_max,
        )

    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }

    @classmethod
    def from_stats_dict(cls, stats_dict: Dict[str, Any]) -> "LogTransformStats":
        instance = cls(
            mean=stats_dict["mean"],
            std=stats_dict["std"],
            range_min=stats_dict.get("range_min", 0.0),
            range_max=stats_dict.get("range_max", 1.0),
        )
        instance.observed_min = stats_dict.get("observed_min")
        instance.observed_max = stats_dict.get("observed_max")
        return instance


class Scale0100Stats(FeatureStats):
    """Scale values in 0-100 range to 0-1."""

    def __init__(
        self,
        scale_factor: float = 100.0,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ):
        self.scale_factor = scale_factor
        self.range_min = range_min
        self.range_max = range_max

    def fit(self, values: pd.Series) -> None:
        # Nothing to compute, we just use the predefined scale factor
        pass

    def transform(self, values: pd.Series) -> pd.Series:
        # Scale to 0-1 range directly
        return values / (self.scale_factor * 100)

    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }

    @classmethod
    def from_stats_dict(cls, stats_dict: Dict[str, Any]) -> "Scale0100Stats":
        return cls(
            scale_factor=stats_dict["scale_factor"],
            range_min=stats_dict.get("range_min", 0.0),
            range_max=stats_dict.get("range_max", 1.0),
        )


class ReturnsStats(FeatureStats):
    """Alternative normalization for returns"""

    def __init__(
        self, scale_factor: float = 1.0, range_min: float = -1.0, range_max: float = 1.0
    ):
        self.scale_factor = scale_factor
        self.range_min = range_min
        self.range_max = range_max
        self.observed_min = None
        self.observed_max = None

    def fit(self, values: pd.Series) -> None:
        # Use 95th percentile as scale factor
        self.scale_factor = float(values.abs().quantile(0.95)) or 1.0

        # Calculate normalized values to determine observed range
        normalized = values / self.scale_factor
        self.observed_min = float(normalized.min())
        self.observed_max = float(normalized.max())

    def transform(self, values: pd.Series) -> pd.Series:
        # First apply returns scaling
        scaled = values / self.scale_factor

        # Then normalize to target range
        return self.range_normalize(
            scaled, self.range_min, self.range_max, self.observed_min, self.observed_max
        )

    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }

    @classmethod
    def from_stats_dict(cls, stats_dict: Dict[str, Any]) -> "ReturnsStats":
        instance = cls(
            scale_factor=stats_dict["scale_factor"],
            range_min=stats_dict.get("range_min", -1.0),
            range_max=stats_dict.get("range_max", 1.0),
        )
        instance.observed_min = stats_dict.get("observed_min")
        instance.observed_max = stats_dict.get("observed_max")
        return instance


FeatureClassifierRule = Tuple[Callable[[str], bool], str]


class FeatureClassifier:
    """
    Classifies features by type based on column names.
    Uses a simpler functional approach with predicates and category names.
    """

    def __init__(self, rules: Optional[List[FeatureClassifierRule]] = None):
        """
        Initialize a feature classifier with the specified rules.

        Args:
            rules: Optional list of rules as (predicate_function, category) tuples.
                  If None, the default rules will be used.
        """
        self.rules = rules or self._get_default_rules()
        self.excluded_metadata = {"time", "timeframe", "asset", "market"}
        self.normalizer_mapping = self._get_default_normalizer_mapping()

    def _get_default_rules(self) -> List[FeatureClassifierRule]:
        """Get the default rules for feature classification."""
        return [
            # OHLC rules
            (lambda col: col in {"open", "high", "low", "close"}, "ohlc"),
            (
                lambda col: col
                in {"open_returns", "high_returns", "low_returns", "close_returns"},
                "ohlc_returns",
            ),
            # Volume rule
            (lambda col: col == "volume", "volume"),
            # Moving Average rules with returns
            (lambda col: col.startswith("MA_") and "returns" in col, "ma_returns"),
            (lambda col: col.startswith("MA_") and "returns" not in col, "ma"),
            (lambda col: col.startswith("EMA_") and "returns" in col, "ema_returns "),
            (lambda col: col.startswith("EMA_") and "returns" not in col, "ema"),
            # Oscillator rules
            (lambda col: col.startswith("RSI_"), "rsi"),
            (lambda col: col.startswith("ADX_"), "adx"),
            (lambda col: col.startswith("MFI_"), "mfi"),
            (lambda col: col.startswith("Stochastic_"), "stochastic"),
            # Bollinger bands
            (
                lambda col: col.startswith("BB_") and "returns" in col,
                "bollinger_bands_returns",
            ),
            (
                lambda col: col.startswith("BB_") and "returns" not in col,
                "bollinger_bands",
            ),
            # MACD
            (lambda col: col.startswith("MACD_") and "returns" in col, "macd_returns"),
            (lambda col: col.startswith("MACD_") and "returns" not in col, "macd"),
            # ATR
            (lambda col: col.startswith("ATR_") and "returns" in col, "atr_returns"),
            (lambda col: col.startswith("ATR_") and "returns" not in col, "atr"),
            # Other indicators
            (lambda col: col.startswith("OBV_"), "obv"),
            (lambda col: col.startswith("CCI_"), "cci"),
            # Special case for return-based features in other category
            (
                lambda col: ("_ret" in col or "_returns" in col)
                and not any(
                    col.startswith(p)
                    for p in [
                        "MA_",
                        "EMA_",
                        "BB_",
                        "MACD_",
                        "ATR_",
                        "RSI_",
                        "ADX_",
                        "MFI_",
                        "OBV_",
                        "CCI_",
                        "Stochastic_",
                    ]
                ),
                "other_returns",
            ),
        ]

    def _get_default_normalizer_mapping(self) -> Dict[str, Any]:
        """Get the default mapping from categories to normalizer types."""
        # Define normalizer types for each category
        return {
            # Return-based indicators should use -1 to 1 range
            "ohlc_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            "ma_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            "ema_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            "macd_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            "atr_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            "bollinger_bands_returns": lambda: ReturnsStats(
                range_min=-1.0, range_max=1.0
            ),
            "other_returns": lambda: ReturnsStats(range_min=-1.0, range_max=1.0),
            # Volume uses log transform and should be 0 to 1
            "volume": lambda: LogTransformStats(range_min=0.0, range_max=1.0),
            # 0-100 range indicators already output in 0-1 range
            "rsi": lambda: Scale0100Stats(range_min=0.0, range_max=1.0),
            "adx": lambda: Scale0100Stats(range_min=0.0, range_max=1.0),
            "mfi": lambda: Scale0100Stats(range_min=0.0, range_max=1.0),
            "stochastic": lambda: Scale0100Stats(range_min=0.0, range_max=1.0),
            # Price-based indicators should use -1 to 1 range
            "ohlc": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
            "ma": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
            "ema": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
            "bollinger_bands": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
            "atr": lambda: ZScoreStats(range_min=0.0, range_max=1.0),  # ATR is positive
            "macd": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
            # Default is Z-score for everything else
            "default": lambda: ZScoreStats(range_min=-1.0, range_max=1.0),
        }

    def classify_feature(self, column_name: str) -> str:
        """
        Classify a single feature based on its column name.

        Args:
            column_name: The column name to classify

        Returns:
            The category the feature belongs to
        """
        # Apply each rule in order until one matches
        for predicate, category in self.rules:
            if predicate(column_name):
                return category

        # If no rule matches and it's not a metadata column, put in "other"
        if column_name not in self.excluded_metadata:
            return "other"

        # Fallback for metadata columns
        return "metadata"

    def classify_features(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features by type based on column names.

        Args:
            column_names: List of column names to categorize

        Returns:
            Dictionary mapping feature types to lists of column names
        """
        # Initialize result dict
        features_mapping: Dict[str, List[str]] = {}

        # Classify each column
        for col in column_names:
            category = self.classify_feature(col)

            # Add category to mapping if not already there
            if category not in features_mapping:
                features_mapping[category] = []

            # Add column to appropriate category
            features_mapping[category].append(col)

        return features_mapping

    def determine_normalizer_type(self, column: str) -> FeatureStats:
        """
        Determine which normalization method to use based on the feature type.

        Args:
            column: Column name

        Returns:
            Instance of FeatureStats to use for this column
        """
        # First, classify the feature
        category = self.classify_feature(column)

        # Look up the normalizer factory in the mapping
        normalizer_factory = self.normalizer_mapping.get(
            category, self.normalizer_mapping["default"]
        )

        # Return an instance
        return normalizer_factory()

    def add_rule(self, predicate: Callable[[str], bool], category: str) -> None:
        """
        Add a new classification rule.

        Args:
            predicate: Function that tests if a column belongs to a category
            category: The category name to assign when predicate returns True
        """
        # Add to the beginning for higher precedence
        self.rules.insert(0, (predicate, category))

    def add_normalizer_mapping(
        self, category: str, normalizer_factory: Callable[[], FeatureStats]
    ) -> None:
        """
        Add or update a mapping from a category to a normalizer factory function.

        Args:
            category: The category name
            normalizer_factory: Function that returns a FeatureStats instance
        """
        self.normalizer_mapping[category] = normalizer_factory


class FeatureNormalizer:
    """
    A class that handles normalization of financial data features.
    Follows the fit/transform paradigm similar to scikit-learn.

    This class supports working with pandas DataFrames only.
    """

    # Normalization methods (kept for backward compatibility)
    Z_SCORE = "z_score"
    LOG_TRANSFORM = "log_transform"
    SCALE_0_100 = "scale_0_100"
    SIGN_PRESERVING = "sign_preserving"

    def __init__(
        self,
        max_samples_for_stats: int = 10000,
        classifier: Optional[FeatureClassifier] = None,
    ):
        """
        Initialize the FeatureNormalizer.

        Args:
            max_samples_for_stats: Maximum number of samples to use when computing normalization stats.
            classifier: Optional custom feature classifier. If None, a default classifier will be created.
        """
        self.max_samples_for_stats = max_samples_for_stats
        self.feature_stats: Dict[str, FeatureStats] = {}
        self.is_fitted = False
        self.classifier = classifier or FeatureClassifier()

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """
        Compute normalization statistics from provided data samples.

        Args:
            df: DataFrame to compute statistics from

        Returns:
            self: The fitted normalizer instance
        """
        # Process each column in the dataframe
        for column in df.columns:
            # Skip metadata columns
            if column in ["time", "timeframe", "asset", "market"]:
                continue

            values = df[column].dropna()

            # Skip if no valid values
            if len(values) == 0:
                continue

            # Determine normalization method based on feature type and create instance
            stats_instance = self.classifier.determine_normalizer_type(column)
            stats_instance.fit(values)
            self.feature_stats[column] = stats_instance

        self.is_fitted = True

        # Print summary
        print(f"Fitted normalizer with {len(self.feature_stats)} features")
        method_counts = {}
        for column, stats in self.feature_stats.items():
            method_name = stats.__class__.__name__
            range_info = f"[{stats.range_min},{stats.range_max}]"
            method_key = f"{method_name} {range_info}"
            method_counts[method_key] = method_counts.get(method_key, 0) + 1

        for method, count in method_counts.items():
            print(f"  - {method}: {count} features")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to input DataFrame using computed statistics.

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            return df

        # Create a copy of the input DataFrame to avoid modifying the original
        df_normalized = df.copy()

        # Apply normalization for each column
        for column, stats in self.feature_stats.items():
            if column in df_normalized.columns:
                df_normalized[column] = stats.transform(df_normalized[column])

        # Handle any remaining NaN values
        df_normalized = df_normalized.fillna(0)

        return df_normalized

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics and apply normalization in one step.

        Args:
            df: DataFrame to compute statistics from and normalize

        Returns:
            Normalized DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save normalization statistics to a JSON file.

        Args:
            path: Path to save the stats file
            metadata: Optional additional metadata to include in the saved file
        """
        if not self.is_fitted:
            raise ValueError("Normalizer is not fitted. Call fit() first.")

        # Convert stats to a dict that can be serialized to JSON
        serializable_stats = {}
        for column, stats in self.feature_stats.items():
            serializable_stats[column] = {
                "method": stats.__class__.__name__,
                "stats": self._convert_numpy(stats.get_stats_dict()),
            }

        # Add metadata
        result_stats = {"stats": serializable_stats}
        if metadata:
            result_stats["metadata"] = metadata

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Save to file
        with open(path, "w") as f:
            json.dump(result_stats, f, indent=2)

        print(f"Normalization stats saved to {path}")

    def _convert_numpy(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        return obj

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """
        Load normalization statistics from a JSON file.

        Args:
            path: Path to the stats file

        Returns:
            Initialized FeatureNormalizer with loaded statistics
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalization stats file not found: {path}")

        try:
            with open(path, "r") as f:
                loaded_data = json.load(f)

            # Create a new normalizer instance
            normalizer = cls()

            # Convert the loaded data back to FeatureStats instances
            if "stats" in loaded_data:
                for column, data in loaded_data["stats"].items():
                    method = data["method"]
                    stats_dict = data["stats"]

                    # Map old method names to new class names for backward compatibility
                    if method == "z_score":
                        normalizer.feature_stats[column] = ZScoreStats.from_stats_dict(
                            stats_dict
                        )
                    elif method == "log_transform":
                        normalizer.feature_stats[column] = (
                            LogTransformStats.from_stats_dict(stats_dict)
                        )
                    elif method == "scale_0_100":
                        normalizer.feature_stats[column] = (
                            Scale0100Stats.from_stats_dict(stats_dict)
                        )
                    elif method == "sign_preserving":
                        normalizer.feature_stats[column] = (
                            SignPreservingStats.from_stats_dict(stats_dict)
                        )
                    else:
                        # Try to load using class name directly
                        stats_class = globals().get(method)
                        if stats_class and issubclass(stats_class, FeatureStats):
                            normalizer.feature_stats[column] = (
                                stats_class.from_stats_dict(stats_dict)
                            )
                        else:
                            raise ValueError(f"Unknown normalization method: {method}")

            normalizer.is_fitted = True

            # Print some metadata if available
            if "metadata" in loaded_data:
                meta = loaded_data["metadata"]
                print(f"Loaded normalization stats from {path}")
                print(f"  - Created: {meta.get('creation_date', 'unknown')}")
                print(f"  - Currency pairs: {meta.get('currency_pairs', 'unknown')}")
                print(f"  - Timeframe: {meta.get('timeframe', 'unknown')}")
                print(f"  - Date range: {meta.get('date_range', 'unknown')}")

            return normalizer
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing normalization stats file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading normalization stats: {e}")
