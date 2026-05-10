import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class PreprocessingService:
    def __init__(self):
        self.encoders = {}

    def clean_data(self, df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        drop_existing = [col for col in drop_columns if col in df.columns]
        df = df.drop(columns=drop_existing)
        df = df.fillna(0)
        return df

    def encode_categoricals(self, df: pd.DataFrame, exclude_columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col in exclude_columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == "object":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            self.encoders[target_col] = target_encoder

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def prepare_dataset_for_multiclass(self, df, target_col='type'):
        """
        Prepara el dataset para clasificación multiclase.
        Elimina columnas que pueden generar fuga de información
        o que no aportan a la generalización del modelo.
        """

        drop_columns = [
            'label',
            'uid',
            'ts',
            'src_port',
            'dst_port'
        ]

        columns_to_drop = drop_columns + [target_col]

        X = df.drop(columns=columns_to_drop, errors='ignore').copy()
        y = df[target_col].copy()

        return X, y
        
    def prepare_dataset_for_binary(self, df, target_col='label'):
        drop_columns = ['type', 'uid', 'ts', 'src_port', 'dst_port']
        X = df.drop(columns=drop_columns + [target_col], errors='ignore').copy()
        y = df[target_col].copy()
        return X, y