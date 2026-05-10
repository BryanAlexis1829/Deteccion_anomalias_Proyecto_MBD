import pandas as pd

class IngestionService:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df

    def validate_columns(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")