import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PreprocessingService:
    def __init__(self):
        self.encoders = {}
        self.imputation_values = {}

    def generate_null_report(
        self,
        df: pd.DataFrame,
        output_path: str | None = None
    ) -> pd.DataFrame:
        """
        Genera un reporte de valores nulos por variable.

        Antes del conteo, convierte "-" a NaN para tratarlo
        como dato faltante.
        """
        df = df.copy()

        # Convertir "-" a nulo
        df = df.replace("-", np.nan)

        total_rows = len(df)

        null_report = pd.DataFrame({
            "variable": df.columns,
            "tipo_dato": df.dtypes.astype(str).values,
            "cantidad_nulos": df.isnull().sum().values,
            "porcentaje_nulos": (df.isnull().sum().values / total_rows) * 100
        })

        null_report = null_report.sort_values(
            by="porcentaje_nulos",
            ascending=False
        )

        if output_path is not None:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            null_report.to_excel(output_path, index=False)

        return null_report

    def get_high_null_columns(
        self,
        df: pd.DataFrame,
        threshold_pct: float = 95.0,
        exclude_columns: list[str] | None = None
    ) -> list[str]:
        """
        Devuelve una lista de columnas cuyo porcentaje de nulos
        es mayor al umbral indicado.

        También considera el valor "-" como faltante.
        """
        df = df.copy()
        df = df.replace("-", np.nan)

        if exclude_columns is None:
            exclude_columns = []

        null_pct = (df.isnull().sum() / len(df)) * 100

        cols_to_drop = [
            col for col, pct in null_pct.items()
            if pct > threshold_pct and col not in exclude_columns
        ]

        return cols_to_drop

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        exclude_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Imputa valores faltantes según el tipo de variable:

        - Variables numéricas: NaN y "-" -> 0
        - Variables categóricas: NaN y "-" -> "missing"
        """
        df = df.copy()

        if exclude_columns is None:
            exclude_columns = []

        # Convertir "-" a nulo antes de imputar
        df = df.replace("-", np.nan)

        # Reiniciar resumen de imputación
        self.imputation_values = {}

        for col in df.columns:
            if col in exclude_columns:
                continue

            total_nulls = df[col].isnull().sum()

            if total_nulls == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
                self.imputation_values[col] = 0
            else:
                df[col] = df[col].fillna("missing")
                self.imputation_values[col] = "missing"

        return df

    def export_imputation_summary(
        self,
        df_before: pd.DataFrame,
        output_path: str
    ) -> pd.DataFrame:
        """
        Exporta un resumen con el criterio de imputación
        utilizado para cada variable.
        """
        df_before = df_before.copy()
        df_before = df_before.replace("-", np.nan)

        resumen = []

        for col, valor_imputado in self.imputation_values.items():
            if pd.api.types.is_numeric_dtype(df_before[col]):
                descripcion = 'Variable numérica: NaN y "-" imputados con 0'
            else:
                descripcion = 'Variable categórica: NaN y "-" imputados con "missing"'

            resumen.append({
                "variable": col,
                "tipo_dato": str(df_before[col].dtype),
                "valor_imputado": valor_imputado,
                "descripcion_tratamiento": descripcion
            })

        imputation_summary = pd.DataFrame(resumen)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        imputation_summary.to_excel(output_path, index=False)

        return imputation_summary

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_columns: list[str],
        null_report_path: str | None = None,
        imputation_summary_path: str | None = None,
        exclude_imputation_columns: list[str] | None = None,
        high_null_threshold: float | None = None,
        exclude_high_null_columns: list[str] | None = None
    ):
        """
        Limpieza general del dataset.

        Pasos:
        1. Genera reporte completo de nulos considerando "-" como faltante.
        2. Identifica columnas con alta nulidad.
        3. Elimina columnas con más de cierto porcentaje de nulos.
        4. Imputa faltantes:
           - numéricas -> 0
           - categóricas -> "missing"
        5. Elimina columnas adicionales definidas por leakage o no relevantes.

        Retorna:
        - df limpio
        - null_report completo
        - lista de columnas eliminadas por alta nulidad
        """
        df = df.copy()

        # Copia original para documentar nulos
        df_before_imputation = df.copy()

        # Generar reporte COMPLETO de nulos
        null_report = self.generate_null_report(
            df=df_before_imputation,
            output_path=null_report_path
        )

        # Detectar columnas con alta nulidad
        high_null_cols = []
        if high_null_threshold is not None:
            high_null_cols = self.get_high_null_columns(
                df=df_before_imputation,
                threshold_pct=high_null_threshold,
                exclude_columns=exclude_high_null_columns
            )

            if high_null_cols:
                print(f"Columnas eliminadas por tener más de {high_null_threshold}% de nulos:")
                print(high_null_cols)

                df = df.drop(columns=high_null_cols, errors="ignore")

        # Imputación
        df = self.impute_missing_values(
            df=df,
            exclude_columns=exclude_imputation_columns
        )

        # Exportar resumen de imputación solo sobre columnas que quedaron
        if imputation_summary_path is not None:
            self.export_imputation_summary(
                df_before=df_before_imputation.drop(columns=high_null_cols, errors="ignore"),
                output_path=imputation_summary_path
            )

        # Eliminar columnas adicionales por leakage
        drop_existing = [
            col for col in drop_columns
            if col in df.columns
        ]

        df = df.drop(columns=drop_existing)

        return df, null_report, high_null_cols

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        exclude_columns: list[str]
    ) -> pd.DataFrame:
        """
        Codifica variables categóricas mediante LabelEncoder.

        Las columnas indicadas en exclude_columns no se codifican.
        """
        df = df.copy()

        categorical_cols = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns

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
        """
        Divide el dataset en entrenamiento y prueba.

        Si la variable objetivo es categórica, se codifica con LabelEncoder.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if str(y.dtype) in ["object", "string", "category"]:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            self.encoders[target_col] = target_encoder

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def prepare_dataset_for_multiclass(
        self,
        df: pd.DataFrame,
        target_col: str = "type"
    ):
        """
        Prepara el dataset para clasificación multiclase.

        Se eliminan columnas que pueden generar fuga de información
        o que no aportan a la generalización del modelo.
        """
        drop_columns = [
            "label",
            "uid",
            "ts",
            "src_port",
            "dst_port"
        ]

        columns_to_drop = drop_columns + [target_col]

        X = df.drop(
            columns=columns_to_drop,
            errors="ignore"
        ).copy()

        y = df[target_col].copy()

        return X, y

    def prepare_dataset_for_binary(
        self,
        df: pd.DataFrame,
        target_col: str = "label"
    ):
        """
        Prepara el dataset para clasificación binaria.

        Se elimina la variable type para evitar fuga de información,
        además de columnas no recomendadas como uid, ts y puertos.
        """
        drop_columns = [
            "type",
            "uid",
            "ts",
            "src_port",
            "dst_port"
        ]

        X = df.drop(
            columns=drop_columns + [target_col],
            errors="ignore"
        ).copy()

        y = df[target_col].copy()

        return X, y