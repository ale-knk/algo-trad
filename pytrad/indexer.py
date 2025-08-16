from typing import List, Optional, Union

import torch


class AssetIndexer:
    """Maps assets to integer indices for use with embeddings in models."""

    def __init__(self, assets: Optional[List[str]] = None):
        # Default assets if none provided
        self.assets = assets or [
            # FOREX
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "EURCHF",
            "EURAUD",
            "EURCAD",
            "EURNZD",
            "GBPJPY",
            "GBPCHF",
            "GBPAUD",
            "GBPCAD",
            "GBPNZD",
            "CHFJPY",
            "AUDJPY",
            "CADJPY",
            "NZDJPY",
            "AUDCHF",
            "CADCHF",
            "NZDCHF",
            "AUDCAD",
            "AUDNZD",
            "CADNZD",
            # CRYPTO
            "BTCUSDT",
            "ETHUSDT",
            "XRPUSDT",
            # STOCK
            "AAPL",
            "GOOG",
            "MSFT",
        ]

        # Create mapping from asset to index
        self.asset_to_index = {asset: idx for idx, asset in enumerate(self.assets)}
        self.index_to_asset = {idx: asset for asset, idx in self.asset_to_index.items()}

        # Number of unique assets
        self.vocab_size = len(self.assets)

    def encode(self, asset: str) -> int:
        """Convert an asset to an integer index"""
        if asset not in self.asset_to_index:
            # Return an index for unknown assets
            return len(self.assets)

        return self.asset_to_index[asset]

    def decode(self, index: int) -> str:
        """Convert an integer index back to an asset"""
        if index not in self.index_to_asset:
            return "unknown"

        return self.index_to_asset[index]

    def batch_encode(self, assets: List[str]) -> torch.Tensor:
        """Convert a list of assets to a tensor of integer indices"""
        indices = [self.encode(asset) for asset in assets]
        return torch.tensor(indices, dtype=torch.long)

    def batch_decode(self, indices: Union[torch.Tensor, List[int]]) -> List[str]:
        """Convert a tensor or list of integer indices back to assets"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        return [self.decode(idx) for idx in indices]

    def __len__(self):
        return self.vocab_size


class MarketIndexer:
    """Maps markets to integer indices for use with embeddings in models."""

    def __init__(self, markets: Optional[List[str]] = None):
        # Default markets if none provided
        self.markets = markets or [
            "forex",
            "crypto",
            "stock",
            "futures",
            "options",
            "bonds",
            "commodities",
            "indices",
            "etf",
            "mutual_fund",
        ]

        # Create mapping from market to index
        self.market_to_index = {market: idx for idx, market in enumerate(self.markets)}
        self.index_to_market = {
            idx: market for market, idx in self.market_to_index.items()
        }

        # Number of unique markets
        self.vocab_size = len(self.markets)

    def encode(self, market: str) -> int:
        """Convert a market to an integer index"""
        if market not in self.market_to_index:
            # Return an index for unknown markets
            return len(self.markets)

        return self.market_to_index[market]

    def decode(self, index: int) -> str:
        """Convert an integer index back to a market"""
        if index not in self.index_to_market:
            return "unknown"

        return self.index_to_market[index]

    def batch_encode(self, markets: List[str]) -> torch.Tensor:
        """Convert a list of markets to a tensor of integer indices"""
        indices = [self.encode(market) for market in markets]
        return torch.tensor(indices, dtype=torch.long)

    def batch_decode(self, indices: Union[torch.Tensor, List[int]]) -> List[str]:
        """Convert a tensor or list of integer indices back to markets"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        return [self.decode(idx) for idx in indices]

    def __len__(self):
        return self.vocab_size


class TimeframeIndexer:
    """Maps timeframes to integer indices for use with embeddings in models."""

    def __init__(self, timeframes: Optional[List[str]] = None):
        # Default timeframes if none provided
        self.timeframes = timeframes or [
            "M1",
            "M5",
            "M15",
            "M30",
            "H1",
            "H4",
            "D1",
            "W1",
            "MN1",
        ]

        # Create mapping from timeframe to index
        self.timeframe_to_index = {
            timeframe: idx for idx, timeframe in enumerate(self.timeframes)
        }
        self.index_to_timeframe = {
            idx: timeframe for timeframe, idx in self.timeframe_to_index.items()
        }

        # Number of unique timeframes
        self.vocab_size = len(self.timeframes)

    def encode(self, timeframe: str) -> int:
        """Convert a timeframe to an integer index"""
        if timeframe not in self.timeframe_to_index:
            # Return an index for unknown timeframes
            return len(self.timeframes)

        return self.timeframe_to_index[timeframe]

    def decode(self, index: int) -> str:
        """Convert an integer index back to a timeframe"""
        if index not in self.index_to_timeframe:
            return "unknown"

        return self.index_to_timeframe[index]

    def batch_encode(self, timeframes: List[str]) -> torch.Tensor:
        """Convert a list of timeframes to a tensor of integer indices"""
        indices = [self.encode(timeframe) for timeframe in timeframes]
        return torch.tensor(indices, dtype=torch.long)

    def batch_decode(self, indices: Union[torch.Tensor, List[int]]) -> List[str]:
        """Convert a tensor or list of integer indices back to timeframes"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        return [self.decode(idx) for idx in indices]

    def __len__(self):
        return self.vocab_size

    def is_shorter_than(self, timeframe1: str, timeframe2: str) -> bool:
        """
        Compares two timeframes and returns True if timeframe1 is shorter than timeframe2.

        Args:
            timeframe1: First timeframe to compare
            timeframe2: Second timeframe to compare

        Returns:
            bool: True if timeframe1 is shorter than timeframe2
        """
        try:
            idx1 = self.encode(timeframe1)
            idx2 = self.encode(timeframe2)
            return idx1 < idx2
        except KeyError:
            raise ValueError(f"Unknown timeframe: {timeframe1} or {timeframe2}")
