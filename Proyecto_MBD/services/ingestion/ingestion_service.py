import os
import glob
import pandas as pd


class IngestionService:
    def __init__(self, data_path: str | None = None):
        self.data_path = data_path

    # ==========================================
    # Un solo archivo CSV
    # ==========================================
    def load_data(self) -> pd.DataFrame:
        if not self.data_path:
            raise ValueError("data_path no fue proporcionado.")
        df = pd.read_csv(self.data_path)
        return df

    def validate_columns(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    # ==========================================
    # Múltiples CSV
    # ==========================================
    def list_csv_files(self, folder_path: str) -> list[str]:
        files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en: {folder_path}")
        return files

    def load_multiple_csv(self, folder_path: str) -> pd.DataFrame:
        files = self.list_csv_files(folder_path)

        df_list = []
        for file in files:
            print(f"Cargando: {file}")
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)

        df = pd.concat(df_list, ignore_index=True)
        return df

    def load_multiple_csv_sample(self, folder_path: str, sample_frac: float = 0.05) -> pd.DataFrame:
        files = self.list_csv_files(folder_path)

        df_list = []
        for file in files:
            print(f"Cargando muestra de: {file}")
            temp_df = pd.read_csv(file)

            # evita errores si el archivo es muy pequeño
            if 0 < sample_frac < 1:
                temp_df = temp_df.sample(frac=sample_frac, random_state=42)

            df_list.append(temp_df)

        df = pd.concat(df_list, ignore_index=True)
        return df

    def load_multiple_csv_with_chunks(self, folder_path: str, chunksize: int = 100000) -> pd.DataFrame:
        files = self.list_csv_files(folder_path)

        df_list = []
        for file in files:
            print(f"Cargando por chunks: {file}")
            chunk_iter = pd.read_csv(file, chunksize=chunksize)

            for chunk in chunk_iter:
                df_list.append(chunk)

        df = pd.concat(df_list, ignore_index=True)
        return df

    # ==========================================
    # Persistencia
    # ==========================================
    def save_dataframe_parquet(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)

    def save_dataframe_csv(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)