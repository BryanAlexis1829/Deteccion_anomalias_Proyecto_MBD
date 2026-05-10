# Proyecto MBD — Network Intrusion Detection System

Pipeline de machine learning para detección de intrusiones en redes, con clasificación **binaria** (normal vs. ataque) y **multiclase** (10 tipos de ataque) sobre un dataset de ~22 millones de registros de tráfico de red.

---

## Tabla de contenidos

- [Descripción del proyecto](#descripción-del-proyecto)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Dataset](#dataset)
- [Requisitos de hardware recomendados](#requisitos-de-hardware-recomendados)
- [Configuración del entorno](#configuración-del-entorno)
- [Pipeline de ejecución](#pipeline-de-ejecución)
- [Modelos implementados](#modelos-implementados)
- [Outputs generados](#outputs-generados)

---

## Descripción del proyecto

El sistema analiza tráfico de red capturado en 23 archivos CSV (formato Zeek/Bro) para clasificar conexiones como benignas o maliciosas. Se implementan modelos ensamblados de árboles de decisión y redes neuronales densas, con análisis de interpretabilidad mediante SHAP values.

**Clases objetivo (multiclase):**
`backdoor`, `ddos`, `dos`, `injection`, `mitm`, `normal`, `password`, `ransomware`, `scanning`, `xss`

---

## Stack tecnológico

| Categoría | Tecnología | Uso |
|---|---|---|
| **Lenguaje** | Python 3.11 | Runtime principal |
| **Entorno** | venv | Aislamiento de dependencias |
| **Notebooks** | Jupyter Notebook 7 | Entorno de ejecución del pipeline |
| **Procesamiento de datos** | Pandas, NumPy, SciPy | Manipulación, análisis y operaciones numéricas |
| **Almacenamiento columnar** | PyArrow (Parquet) | Persistencia eficiente del dataset consolidado |
| **ML — árboles** | scikit-learn, XGBoost, LightGBM | Random Forest, gradient boosting binario y multiclase |
| **ML — redes neuronales** | TensorFlow / Keras | Redes neuronales densas con EarlyStopping |
| **Explicabilidad** | SHAP | Importancia de features por SHAP values |
| **Visualización** | Matplotlib, Seaborn | Gráficos de métricas, matrices de confusión, SHAP plots |
| **Persistencia de modelos** | joblib | Serialización de modelos sklearn, scalers y encoders |
| **Reportes** | openpyxl | Exportación de reportes de nulos e imputación a Excel |
| **SO recomendado** | Linux (Debian 12 / Ubuntu 22.04) | Compatibilidad óptima con TensorFlow y LightGBM |

---

## Estructura del repositorio

```
Proyecto_MBD/
├── full_notebooks/                          # Flujo principal de ejecución
│   ├── 01_eda_full.ipynb                    # Análisis exploratorio (muestra 5%)
│   ├── 02_preprocessing_full.ipynb          # Limpieza, imputación y encoding
│   ├── 03_training_binary_full.ipynb        # Entrenamiento RF / XGB / LGB — binario
│   ├── 03_training_multiclass_full.ipynb    # Entrenamiento RF / XGB / LGB — multiclase
│   ├── 04_evaluation_full.ipynb             # Evaluación y métricas
│   ├── 05_training_dnn_multiclass_full.ipynb # Red neuronal densa — multiclase
│   ├── 06_training_dnn_binary_full.ipynb     # Red neuronal densa — binaria
│   ├── 07_shap_analysis_full.ipynb           # Importancia de features con SHAP
│   └── 08_results_comparison.ipynb           # Comparación global de todos los modelos
│
├── services/
│   ├── ingestion/
│   │   └── ingestion_service.py     # Carga de CSVs (individual, batch, chunked)
│   ├── preprocessing/
│   │   └── preprocessing_service.py # Limpieza, imputación, encoding, splits
│   └── training/
│       └── training_service.py      # Entrenamiento RF, XGBoost, LightGBM
│
├── utils/
│   └── config.py                    # Rutas, hiperparámetros y constantes globales
│
├── data/
│   ├── raw/                         # CSVs fuente sin modificar (Network_dataset_1.csv … _23.csv)
│   ├── processed/                   # Parquets consolidados generados por el pipeline
│   └── artifacts/
│       └── full/                    # Modelos, scalers, encoders, predicciones y métricas
│
├── images/                          # Visualizaciones generadas automáticamente
├── requirements.txt
└── README.md
```

---

## Dataset

| Propiedad | Detalle |
|---|---|
| Archivos fuente | 23 CSVs — `Network_dataset_1.csv` … `Network_dataset_23.csv` |
| Features | 44 columnas: IP, puertos, protocolo, flags DNS/SSL/HTTP, bytes, duración, etc. |
| Target binario | `label` — `0`: normal · `1`: ataque |
| Target multiclase | `type` — 10 categorías de ataque |
| Registros totales | ~22,338,021 |
| Muestra EDA | 5% (~1.1M registros) para análisis exploratorio |
| Muestra SHAP | 1,000 registros para análisis de interpretabilidad |

Los 23 CSVs se consolidan durante el paso de preprocessing en un único Parquet (`network_full_consolidated.parquet`). Los pasos posteriores leen únicamente ese archivo.

---

## Requisitos de hardware recomendados

El pipeline opera sobre ~22M registros. Para garantizar tiempos de ejecución razonables se recomienda:

### Google Cloud — `e2-highmem-8`

| Recurso | Especificación |
|---|---|
| **Tipo de máquina** | `e2-highmem-8` |
| **vCPUs** | 8 |
| **Memoria RAM** | 64 GB |
| **Disco** | SSD persistente ≥ 100 GB |
| **Sistema operativo** | Debian 12 / Ubuntu 22.04 LTS |
| **Python** | 3.11 |

**Por qué estos recursos:**
- El Parquet consolidado ocupa varios GB en RAM al cargarse con Pandas.
- XGBoost y LightGBM aprovechan los 8 cores para entrenamiento paralelo.
- TensorFlow necesita al menos 8 GB libres para entrenar la DNN sobre el split completo.
- El análisis SHAP sobre modelos de árbol entrenados con millones de registros es intensivo en memoria.

---

## Configuración del entorno

> Se recomienda ejecutar en **Linux (Debian 12 / Ubuntu 22.04)**. TensorFlow y LightGBM tienen soporte nativo y mejor rendimiento en Linux que en otros sistemas operativos.

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd Proyecto_MBD
```

### 2. Instalar Python 3.11 (si no está disponible)

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 3. Crear y activar el entorno virtual

```bash
python3.11 -m venv venv
source venv/bin/activate
```

> El entorno virtual debe estar **siempre activo** antes de ejecutar cualquier comando del proyecto.

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Registrar el kernel de Jupyter

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

### 6. Colocar los datos fuente

Copiar los 23 archivos CSV dentro de `data/raw/`:

```
data/raw/
├── Network_dataset_1.csv
├── Network_dataset_2.csv
│   ...
└── Network_dataset_23.csv
```

### 7. Lanzar Jupyter

```bash
jupyter notebook
```

> Al abrir cualquier notebook seleccionar el kernel **"Python (venv)"**. El módulo `utils/config.py` crea automáticamente los directorios necesarios al importarse — no hace falta crearlos manualmente.

---

## Pipeline de ejecución

Ejecutar los notebooks de `full_notebooks/` **en el siguiente orden**. Cada notebook produce artefactos que el siguiente consume.

```
01_eda_full.ipynb
  └─ Lee los 23 CSVs mediante IngestionService
  └─ Genera estadísticas descriptivas, distribuciones y correlaciones sobre muestra del 5%
  └─ Guarda visualizaciones en images/

02_preprocessing_full.ipynb
  └─ Consolida los 23 CSVs → data/processed/network_full_consolidated.parquet
  └─ Detecta y reporta nulos → reporte_nulos_dataset_full.xlsx
  └─ Imputa, limpia y codifica variables categóricas (LabelEncoder)
  └─ Genera splits estratificados binario y multiclase → data/artifacts/full/

03_training_binary_full.ipynb
  └─ Carga X_train_full_binary.parquet / X_test_full_binary.parquet
  └─ Entrena Random Forest, XGBoost, LightGBM con class_weight='balanced'
  └─ Guarda → rf_binary_full.pkl, xgb_binary_full.pkl, lgb_binary_full.pkl

03_training_multiclass_full.ipynb
  └─ Mismo flujo para clasificación multiclase (10 clases)
  └─ Guarda → rf_multiclass_full.pkl, xgb_multiclass_full.pkl, lgb_multiclass_full.pkl

04_evaluation_full.ipynb
  └─ Calcula accuracy, F1-score, ROC-AUC, MCC y matrices de confusión para los 6 modelos
  └─ Exporta → binary_model_results_full.csv, multiclass_model_results_full.csv
  └─ Guarda visualizaciones de métricas en images/

05_training_dnn_multiclass_full.ipynb
  └─ Entrena red neuronal densa (Dense 128→64→32) para multiclase con EarlyStopping
  └─ Guarda modelo → dnn_multiclass_full.keras

06_training_dnn_binary_full.ipynb
  └─ Mismo flujo para clasificación binaria
  └─ Guarda modelo → dnn_binary_full.keras

07_shap_analysis_full.ipynb
  └─ Calcula SHAP values sobre muestra de 1,000 registros con el modelo XGBoost
  └─ Exporta ranking de importancia → shap_feature_importance_full_xgb.csv
  └─ Guarda gráficos de top-15 features en images/

08_results_comparison.ipynb
  └─ Consolida resultados de todos los modelos (árbol + DNN, binario + multiclase)
  └─ Genera tabla comparativa y visualizaciones finales
  └─ Exporta → global_results_summary.csv, evaluation_summary_full.csv
```

---

## Modelos implementados

| Modelo | Biblioteca | Clasificación | Archivo guardado |
|---|---|---|---|
| Random Forest | scikit-learn | Binaria y multiclase | `rf_*_full.pkl` |
| XGBoost | xgboost | Binaria y multiclase | `xgb_*_full.pkl` |
| LightGBM | lightgbm | Binaria y multiclase | `lgb_*_full.pkl` |
| Red neuronal densa | TensorFlow / Keras | Binaria y multiclase | `dnn_*_full.keras` |

Arquitectura DNN: `Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Output`

Todos los modelos de árbol usan `class_weight='balanced'` para compensar el desbalance entre clases.

---

## Outputs generados

Al finalizar el pipeline completo se habrán producido los siguientes artefactos en `data/artifacts/full/`:

| Categoría | Archivos |
|---|---|
| Modelos entrenados | `rf_binary_full.pkl`, `xgb_binary_full.pkl`, `lgb_binary_full.pkl`, `rf_multiclass_full.pkl`, `xgb_multiclass_full.pkl`, `lgb_multiclass_full.pkl`, `dnn_binary_full.keras`, `dnn_multiclass_full.keras` |
| Scalers | `scaler_binary_full.pkl`, `scaler_multiclass_full.pkl` |
| Encoders | `feature_encoders_full_binary.pkl`, `feature_encoders_full_multiclass.pkl`, `target_encoder_full_multiclass.pkl` |
| Splits de datos | `X_train_full_*.parquet`, `X_test_full_*.parquet`, `y_train_full_*.csv`, `y_test_full_*.csv` |
| Métricas | `binary_model_results_full.csv`, `multiclass_model_results_full.csv`, `global_results_summary.csv`, `evaluation_summary_full.csv` |
| Predicciones | `*_predictions_full.csv` (un archivo por modelo) |
| SHAP | `shap_feature_importance_full_xgb.csv` |
| Reportes de calidad | `reporte_nulos_dataset_full.xlsx`, `resumen_imputacion_dataset_full.xlsx` |
| Visualizaciones | `images/` — matrices de confusión, comparativas de métricas, SHAP top-15, distribuciones |
