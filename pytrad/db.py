import csv
from datetime import datetime

from pymongo import MongoClient


class MongoDBHandler:
    def __init__(self, host="localhost", port=27017):
        self.client = MongoClient(host, port)
        self.db = self.client["algo_trad"]
        self._indexed_collections = (
            set()
        )  # Para llevar registro de colecciones indexadas

    def _create_indexes(self):
        """Crea los índices necesarios para optimizar las consultas si no existen."""
        if self.collection.name in self._indexed_collections:
            return  

        existing_indexes = self.collection.index_information()

        if "time_1" not in existing_indexes:
            self.collection.create_index([("time", 1)])
        
        if "timeframe_1" not in existing_indexes:
            self.collection.create_index([("timeframe", 1)])

        if "timeframe_1_time_1" not in existing_indexes:
            self.collection.create_index([("timeframe", 1), ("time", 1)])

        self._indexed_collections.add(self.collection.name)

    def set_collection(self, collection_name: str):
        self.collection = self.db[collection_name]
        self._create_indexes()  # Solo creará índices si es necesario

    def insert_candle(self, candle_data: dict):
        """Inserta un solo registro de vela en la colección."""
        self.collection.insert_one(candle_data)

    def insert_candles_from_csv(
        self,
        csv_path: str,
        date_format="%Y-%m-%d %H:%M:%S",
        asset: str = "EURUSD",
        market: str = "FOREX",
        timeframe: str = "M15",
    ):
        """
        Lee un CSV y pobla la colección con los datos de las velas.
        Se espera que el CSV tenga las columnas: timestamp, open, high, low, close, volume.
        """
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            bulk_data = []
            for row in reader:
                try:
                    candle = {
                        "time": datetime.strptime(row["Time"], date_format),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                        "asset": asset,
                        "market": market,
                        "timeframe": timeframe,
                    }
                    bulk_data.append(candle)
                except Exception as e:
                    print(f"Error procesando fila {row}: {e}")

            if bulk_data:
                self.collection.insert_many(bulk_data)
                print(
                    f"Insertados {len(bulk_data)} registros en {self.collection.name}."
                )
            else:
                print("No se insertó ningún registro.")


if __name__ == "__main__":
    assets_dict = {
        "forex": ["EURUSD", "GBPUSD", "USDCAD"],
    }
    db_handler = MongoDBHandler()
    for market in assets_dict.keys():
        for asset in assets_dict[market]:
            for timeframe in ["M15", "H1"]:
                filepath = f"/Users/alejandro.requena/Kanka/projects/algo-trad/data/{asset}_{timeframe}.csv"
                db_handler.set_collection(asset)
                db_handler.insert_candles_from_csv(
                    filepath,
                    date_format="%Y-%m-%d %H:%M:%S",
                    asset=asset,
                    market=market,
                    timeframe=timeframe,
                )
