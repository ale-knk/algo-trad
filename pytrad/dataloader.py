from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import torch

from pytrad.dataset import MultiWindowDataset
from pytrad.indexer import TimeframeIndexer


class MultiWindowDataloader:
    """
    Dataloader para múltiples CandleDatasets con proporciones fijas de assets.
    Avanza de forma lineal en el tiempo y proporciona batches balanceados.
    """

    def __init__(
        self,
        multi_dataset: MultiWindowDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Inicializa el MultiCandleDataloader.

        Args:
            dataset: Instancia de MultiCandleDataset que contiene los datos organizados.
            batch_size: Tamaño total del batch.
            asset_proportions: Diccionario con proporción de cada asset en el batch.
                               Si es None, se usará una distribución uniforme.
            time_step: Incremento de tiempo entre batches. Si es None, se calculará automáticamente.
            shuffle: Si es True, se mezclan las ventanas dentro del mismo punto temporal.
            seed: Semilla para reproducibilidad.
            window_size: Tamaño de cada subventana (número de candles).
            stride: Paso entre subventanas consecutivas.
        """
        self.multi_dataset = multi_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.window_size = self.multi_dataset.window_size
        self.stride = self.multi_dataset.stride

        self.timeframe_indexer = TimeframeIndexer()
        self.timeframes = self._get_all_timeframes()
        self.lower_timeframe = self._find_lower_timeframe()

        self.assets = list(multi_dataset.datasets_kv.keys())

        self.mapping = self._create_mapping()

        self._prepare_iteration()

    def _create_mapping(self):
        current_idx = 0
        mapping = {}
        lower_timeframe = "M15"

        for asset in self.multi_dataset.datasets_kv.keys():
            dataset = self.multi_dataset.datasets_kv[asset][lower_timeframe]
            for idx, end_date in dataset.mapping.items():
                try:
                    for timeframe in self.multi_dataset.datasets_kv[asset].keys():
                        if timeframe != lower_timeframe:
                            window = self.multi_dataset.datasets_kv[asset][
                                timeframe
                            ].window
                            window.get_subwindow_by_datetime(
                                end_date, self.window_size
                            )
                    mapping[current_idx] = (asset, end_date)
                    current_idx += 1
                except ValueError:
                    continue

        return mapping

    def _prepare_iteration(self):
        """
        Prepara un generador de índices para la iteración.
        Si shuffle es True, mezcla los índices antes de crear los batches.
        """
        # Obtener todos los índices del mapping
        indices = list(self.mapping.keys())

        # Aplicar shuffle si está habilitado
        if self.shuffle:
            import random
            random.seed(self.seed)
            random.shuffle(indices)

        # Crear el generador de batches
        def batch_generator():
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

        # Guardar el generador
        self.index_generator = batch_generator()

        # Calcular el número total de batches usando la nueva función
        self.total_batches = self._get_number_of_batches()

        # Inicializar control de iteración
        self.current_batch = 0

    def _get_all_timeframes(self) -> List[str]:
        """Obtiene todos los timeframes disponibles en el dataset."""
        timeframes = set()
        for asset_datasets in self.multi_dataset.datasets_kv.values():
            timeframes.update(asset_datasets.keys())
        return list(timeframes)

    def _get_number_of_batches(self) -> int:
        """
        Calcula el número máximo de batches que se pueden hacer con el mapping actual.

        Returns:
            int: Número total de batches, calculado como ceil(total_items / batch_size)
        """
        total_items = len(self.mapping)
        return (total_items + self.batch_size - 1) // self.batch_size

    def _find_lower_timeframe(self) -> str:
        """Encuentra el timeframe más corto entre los disponibles."""
        if not self.timeframes:
            return "M15"  # Default en caso de no encontrar timeframes

        # Encontrar el timeframe con el índice más bajo (el más corto)
        lower_tf = self.timeframes[0]
        for tf in self.timeframes[1:]:
            if self.timeframe_indexer.is_shorter_than(tf, lower_tf):
                lower_tf = tf

        return lower_tf

    def __len__(self) -> int:
        """Devuelve el número total de batches."""
        return self.total_batches

    def reset(self):
        """Reinicia el iterador."""
        self._prepare_iteration()

    def __iter__(self):
        """Devuelve el iterador."""
        self.reset()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Obtiene el siguiente batch de ventanas agitadas.

        Returns:
            Diccionario con tensores para "ohlc", "ohlc_returns", "indicators", "time_transformed", "asset", "market", "timeframe"
        """
        if self.current_batch >= self.total_batches:
            raise StopIteration

        batch_indexes = next(self.index_generator)
        batch_by_timeframe = {}
        for index in batch_indexes:
            asset, end_date = self.mapping[index]
            windows = self.multi_dataset.get_windows(end_date, asset)

            # Procesar cada ventana (una por timeframe)
            for window in windows:

                tensor_dict = window.to_tensors()
                timeframe_tensor = tensor_dict["timeframe"]
                timeframe = self.timeframe_indexer.decode(
                    timeframe_tensor.item()
                )
                if timeframe not in batch_by_timeframe:
                    batch_by_timeframe[timeframe] = {
                        "ohlc": [],
                        "ohlc_returns": [],
                        "indicators": [],
                        "time": [],
                        "asset": [],
                        "market": [],
                        "timeframe": [],
                    }

                batch_by_timeframe[timeframe]["ohlc"].append(
                    tensor_dict["ohlc"]
                )
                batch_by_timeframe[timeframe]["ohlc_returns"].append(
                    tensor_dict["ohlc_returns"]
                )
                batch_by_timeframe[timeframe]["indicators"].append(
                    tensor_dict["indicators"]
                )
                batch_by_timeframe[timeframe]["time"].append(
                    tensor_dict["time"]
                )

                # Añadir tensores de asset, market y timeframe que ya vienen codificados
                batch_by_timeframe[timeframe]["asset"].append(
                    tensor_dict["asset"]
                )
                batch_by_timeframe[timeframe]["market"].append(
                    tensor_dict["market"]
                )
                batch_by_timeframe[timeframe]["timeframe"].append(
                    tensor_dict["timeframe"]
                )

        result = {}
        for timeframe, data in batch_by_timeframe.items():
            result[timeframe] = {
                "ohlc": torch.stack(data["ohlc"]),
                "ohlc_returns": torch.stack(data["ohlc_returns"]),
                "indicators": torch.stack(data["indicators"]),
                "time": torch.stack(data["time"]),
                "asset": torch.cat(data["asset"]).view(-1, 1),
                "market": torch.cat(data["market"]).view(-1, 1),
                "timeframe": torch.cat(data["timeframe"]).view(-1, 1),
            }

        return result
    
