import csv
from datetime import datetime
from pymongo import MongoClient

class MongoDBHandler:
    def __init__(self, collection_name="EUR_USD_M15", host="localhost", port=27017):
        self.client = MongoClient(host, port)
        self.db = self.client["algo_trad"]
        self.collection = self.db[collection_name]

    def set_collection(self, collection_name: str):
        self.collection = self.db[collection_name]

    def insert_candle(self, candle_data: dict):
        """Inserta un solo registro de vela en la colección."""
        self.collection.insert_one(candle_data)

    def insert_candles_from_csv(self, csv_path: str, date_format="%Y-%m-%d %H:%M:%S"):
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
                        "volume": float(row["Volume"])
                    }
                    bulk_data.append(candle)
                except Exception as e:
                    print(f"Error procesando fila {row}: {e}")

            if bulk_data:
                self.collection.insert_many(bulk_data)
                print(f"Insertados {len(bulk_data)} registros.")
            else:
                print("No se insertó ningún registro.")

if __name__ == "__main__":
    db_handler = MongoDBHandler()
    for currency in ["EURUSD"]:
        for timeframe in ["D1","H1","M30","M15","M5"]:
            filepath = f"/Users/alejandro.requena/Kanka/projects/algo-trad/data/{currency}_{timeframe}.csv"
            db_handler.set_collection(f"{currency}_{timeframe}")
            db_handler.insert_candles_from_csv(filepath, date_format="%Y-%m-%d %H:%M:%S")
    