import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# Rutas base
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed")
ARTIFACTS_PATH = os.path.join(DATA_DIR, "artifacts")

# =========================
# Dataset test (flujo ya funcional)
# =========================
DATA_PATH = os.path.join(RAW_DATA_DIR, "train_test_network.csv")
TRAIN_TEST_PATH = DATA_PATH

# =========================
# Dataset completo (23 CSV)
# =========================
FULL_DATASET_DIR = PROCESSED_PATH

# =========================
# Archivos consolidados del flujo full
# =========================
FULL_CONSOLIDATED_PATH = os.path.join(PROCESSED_PATH, "network_full_consolidated.parquet")
FULL_SAMPLE_PATH = os.path.join(PROCESSED_PATH, "network_full_sample.parquet")

# =========================
# Artefactos flujo actual (train_test)
# =========================
TRAIN_TEST_ARTIFACTS_DIR = os.path.join(ARTIFACTS_PATH, "train_test")

# =========================
# Artefactos flujo full (23 datasets)
# =========================
FULL_ARTIFACTS_DIR = os.path.join(ARTIFACTS_PATH, "full")

# =========================
# Reportes flujo full
# =========================
FULL_NULL_REPORT_PATH = os.path.join(
    FULL_ARTIFACTS_DIR,
    "reporte_nulos_dataset_full.xlsx"
)

FULL_IMPUTATION_SUMMARY_PATH = os.path.join(
    FULL_ARTIFACTS_DIR,
    "resumen_imputacion_dataset_full.xlsx"
)

# =========================
# Configuración de columnas
# =========================
DROP_COLUMNS_BASE = ["src_ip", "dst_ip", "src_port", "dst_port"]
DROP_COLUMNS_OPTIONAL = ["src_port", "dst_port"]

BINARY_TARGET = "label"
MULTICLASS_TARGET = "type"

# =========================
# Parámetros de partición
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# Parámetros de muestreo para EDA/SHAP del flujo full
# =========================
EDA_SAMPLE_FRAC = 0.05
SHAP_SAMPLE_SIZE = 1000

# =========================
# Creación automática de carpetas
# =========================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
os.makedirs(TRAIN_TEST_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FULL_ARTIFACTS_DIR, exist_ok=True)