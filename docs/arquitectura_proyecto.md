A partir de la descripción detallada de la arquitectura de _Machine Learning_ proporcionada, se genera la siguiente estructura de árbol de directorios. Esta estructura sigue las convenciones de los proyectos de ML/DL para separar datos, código fuente (`src`), ejecutables (`scripts`), y artefactos generados (`outputs`, `models`, `reports`), tal como se describe en la arquitectura de un pipeline robusto.

La clase `MLPipeline` es la orquestadora central (invocada por CLI, UI y _notebooks_), y un objeto `Config` centraliza las rutas y variables objetivo.

```
.
├── config/
│   └── config.py                 # Contiene el objeto `Config` centralizando rutas y variables objetivo (ej. 'y')
│
├── data/                         # Carpeta raíz para todos los datos
│   ├── raw/                      # Entrada de datos sin modificar (data/raw en formato CSV)
│   │   └── data_input.csv
│   └── processed/                # Datasets limpios o transformados, listos para Feature Engineering/Modelado (data/processed)
│       └── X_processed.parquet
│
├── src/                          # Código fuente de la lógica interna (importada por scripts, UI y notebooks)
│   ├── __init__.py
│   ├── pipeline.py               # Clase `MLPipeline` (orquestadora de fases: load, explore, preprocess, engineer, train, interpret)
│   ├── eda.py                    # Funciones para `explore_data` (src/eda.py) y generación de resúmenes/gráficos
│   ├── preprocessing/            # Lógica para la fase `preprocess_data` (src/preprocessing/)
│   │   └── clean.py
│   ├── features/                 # Lógica para la fase de *Feature Engineering* (src/features/engineer.py)
│   │   └── engineer.py           # Añade features temporales, agregaciones, ratios, categóricas, género.
│   └── models/                   # Contiene las definiciones de modelos y funciones de entrenamiento/evaluación (src/models/)
│       └── model_architecture.py
│
├── scripts/                      # Scripts ejecutables por CLI para orquestar las fases (scripts/stepX)
│   ├── run_pipeline.py           # Script de ejecución general
│   ├── execute_pipeline.py       # Script de ejecución/segmentación de fases
│   ├── verify_pipeline.py        # Script de verificación de fases o resultados
│   ├── step2_train.py            # Lógica de `Train/evaluate` (referencia a src/models/)
│   ├── step3_evaluate.py         # Lógica de `Train/evaluate` (referencia a src/models/)
│   └── step4_interpretability.py # Interpretabilidad (ej. con SHAP/Permutation Importance)
│
├── ui/                           # Interfaz de usuario (Streamlit UI)
│   ├── app.py                    # Interfaz principal de Streamlit
│   └── pipeline_executor.py      # Lógica de ejecución interactiva que llama a la `MLPipeline`
│
├── notebooks/                    # Entorno para invocación interactiva y prototipado (invoca a MLPipeline)
│   └── full_pipeline_run.ipynb
│
├── outputs/                      # Artefactos de EDA y métricas
│   ├── eda/                      # Artefactos generados por EDA (outputs/eda)
│   │   ├── resumen/              # Resúmenes tabulares (outputs/resumen)
│   │   └── figures/              # Gráficos de distribución/outliers (outputs/figures)
│   └── metrics/                  # Archivos de métricas de rendimiento (F1, MAE, etc.)
│
├── models/                       # Artefactos de modelos (adicional a src/models/)
│   ├── trained/                  # Modelos serializados/guardados (models/trained)
│   └── metadata/                 # Metadatos del modelo (hiperparámetros, esquema de features) (models/metadata)
│
├── reports/                      # Informes finales del proyecto (reports/)
│   └── final_report.pdf
│
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Descripción del proyecto
```

Este es un conjunto de ejemplos representativos para los archivos clave de la arquitectura de _Deep Learning_ y _Machine Learning_ que describió, utilizando como contexto el proyecto de **Modelos Predictivos para la Educación Superior en Chile** (Clasificación de Modalidad y Regresión de Edad Promedio).

---

### 1. Archivos de Configuración

#### `config/config.py`

Este archivo centraliza las variables inmutables, rutas y la definición de las variables objetivo (`y`) y explicativas (`X`).

```
from pathlib import Path

# Rutas Centrales del Proyecto
class Config:
    # Rutas
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_RAW_PATH = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed"
    OUTPUTS_EDA_PATH = BASE_DIR / "outputs" / "eda"
    MODELS_TRAINED_PATH = BASE_DIR / "models" / "trained"

    # Archivos
    RAW_DATA_FILE = DATA_RAW_PATH / "data_input.csv"
    PROCESSED_TRAIN_FILE = DATA_PROCESSED_PATH / "X_train_engineered.parquet"
    MODEL_FILENAME = MODELS_TRAINED_PATH / "mlp_modalidad.pkl"

    # Variables del Modelo
    TARGET_CLASSIFICATION = "MODALIDAD_BIN" # Presencial vs No presencial
    TARGET_REGRESSION = "PROMEDIO_EDAD_PROGRAMA" # Edad continua

    FEATURES_CINE = ['AREA_CINE', 'SUB_AREA_CINE']
    FEATURES_INSTITUCIONALES = ['NIVEL_INSTITUCIONAL', 'INSTITUCION']
    FEATURES_GEOGRAFICAS = ['REGION', 'COMUNA']
    FEATURES_CONTINUAS = ['DURACION_TOTAL_SEMESTRES', 'TAMANO_PROGRAMA']

    # Hiperparámetros iniciales
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    IMBALANCE_RATIO_CRITICAL = 1.5 # Umbral para desbalance
```

### 2. Código Fuente (`src/`)

#### `src/eda.py`

Contiene funciones esenciales para el Análisis Exploratorio de Datos (EDA) y la carga robusta. El siguiente ejemplo muestra la función para cargar datos de forma segura, como se detalla en las fuentes:

```
import pandas as pd
from pathlib import Path
from config.config import Config

def cargar_csv(ruta: str, sep: str = ",", encoding_prioridad=("utf-8", "latin-1")) -> pd.DataFrame:
    """ Lee un archivo CSV probando múltiples 'encodings' y valida la existencia del archivo. """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}") # Verifica existencia

    ultimo_error = None
    for enc in encoding_prioridad:
        try:
            # low_memory=False para evitar tipos 'partidos'
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
            return df
        except Exception as e:
            ultimo_error = e
            continue

    raise RuntimeError(f"No se pudo leer el CSV. Último error: {ultimo_error}")

def explore_data(df: pd.DataFrame, target_col: str):
    # Genera la tabla de calidad de columnas
    resumen = resumen_columnas(df)
    # Genera gráficos de distribución del objetivo
    # ... (código para generar 'outputs/eda/figures/distribucion_obj.png')

    if resumen.loc[target_col, 'pct_missing'] > 0:
        print(f"Advertencia: Target '{target_col}' tiene faltantes. Requiere imputación o descarte.")

# El resto de las funciones de EDA (resumen_columnas, descriptivos, etc.) irían aquí
```

#### `src/preprocessing/clean.py`

Lógica para limpiar y preparar datos, incluyendo el manejo de nulos y la conversión de tipos:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from config.config import Config

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Imputa datos faltantes según la política definida (ej. 0 para edad, mediana para continuas). """

    # 1. Imputar nulos en PROMEDIO_EDAD_PROGRAMA (detectada alta nulidad en rangos etarios)
    # Convertir primero a numérico si está como 'object'
    df[Config.TARGET_REGRESSION] = pd.to_numeric(df[Config.TARGET_REGRESSION], errors='coerce')
    df[Config.TARGET_REGRESSION] = df[Config.TARGET_REGRESSION].fillna(df[Config.TARGET_REGRESSION].median())

    # 2. Imputar features categóricas faltantes con 'MISSING'
    for col in Config.FEATURES_CINE + Config.FEATURES_INSTITUCIONALES:
        df[col] = df[col].fillna('MISSING')

    return df

def apply_scaling(X_train, X_test):
    """ Aplica estandarización (z-score) a variables numéricas, aprendido solo en train. """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    numeric_cols = Config.FEATURES_CONTINUAS

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler

def split_data(df: pd.DataFrame, target: str):
    """ Partición estratificada para mantener proporciones de clases. """
    # En ML tabular, la estratificación es crucial para evitar particiones desbalanceadas

    X = df.drop(columns=[target])
    y = df[target]

    # Split: 80% Train/Val, 20% Test, estratificado por el target de clasificación
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED)

    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test
```

#### `src/features/engineer.py`

Contiene la lógica de _Feature Engineering_ para crear las variables mencionadas (ratios, categóricas, etc.):

```
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Crea nuevas features: ratios, variables categóricas, y codificación. """

    # 1. Variables de género (Si estuvieran disponibles: Titulaciones H/M)
    # df['RATIO_GENERO'] = df['TITULACIONES_MUJERES'] / df['TITULACIONES_HOMBRES']

    # 2. Features temporales (si el dato es una serie de tiempo, usaríamos lags, pero aquí usamos año)
    df['POST_2020'] = (df['ANIO'] >= 2020).astype(int) # Efecto pandemia

    # 3. Codificación de variables categóricas (One-Hot Encoding para MLP tabular)
    df = pd.get_dummies(df, columns=Config.FEATURES_CINE + Config.FEATURES_INSTITUCIONALES, drop_first=True)

    # 4. Agregaciones / Ratios (Ejemplo de complejidad institucional)
    df['DURACION_TASA'] = df['TAMANO_PROGRAMA'] / df['DURACION_TOTAL_SEMESTRES']

    # El set resultante alimenta X_train y X_test engineered.
    return df
```

#### `src/models/model_architecture.py`

Define la estructura de la red neuronal, en este caso, un Perceptrón Multicapa (MLP) adecuado para datos tabulares. Utiliza técnicas de regularización como _Batch Normalization_ y _Dropout_:

```
import tensorflow as tf
from tensorflow.keras import layers, Model
from config.config import Config

def build_mlp_classifier(n_features: int, n_classes: int = 2) -> Model:
    """
    Construye un Perceptrón Multicapa (MLP) para clasificación binaria/multiclase,
    incluyendo Batch Normalization y Dropout para regularización.
    """
    # Arquitectura MLP con capas densas
    inputs = tf.keras.Input(shape=(n_features,))

    # Primera capa Densa + ReLU
    x = layers.Dense(128, activation='relu')(inputs)

    # Batch Normalization (para estabilidad)
    x = layers.BatchNormalization()(x)

    # Dropout (para reducir overfitting)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)

    # Capa de salida: Sigmoid para clasificación binaria (Presencial/No presencial)
    if n_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(x) # Softmax para multiclase

    model = Model(inputs, outputs)
    return model

def compile_and_train(model, train_data, val_data):
    """ Configura optimizador, pérdida y ejecuta el entrenamiento. """

    # El Descenso del Gradiente se implementa mediante optimizadores como Adam
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', # BCE para clasificación binaria
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Early Stopping como regularizador clave
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    print("Iniciando entrenamiento...")
    # Durante fit(), se aplica Backpropagation para ajustar pesos
    history = model.fit(train_data, validation_data=val_data,
                        epochs=200, callbacks=[es])
    return history
```

#### `src/pipeline.py`

Clase orquestadora (`MLPipeline`) que sigue la secuencia de fases descrita en la arquitectura:

```
import pandas as pd
from config.config import Config
from src.eda import cargar_csv, explore_data
from src.preprocessing.clean import handle_missing_data, split_data, apply_scaling
from src.features.engineer import engineer_features
from src.models.model_architecture import build_mlp_classifier, compile_and_train

class MLPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.model = None

    def load_data(self):
        """ Fase 1: Carga robusta de datos crudos. """
        print("1. Cargando datos...")
        self.df = cargar_csv(self.config.RAW_DATA_FILE)
        return self

    def explore_data(self):
        """ Fase 2: Análisis Exploratorio (EDA). """
        print("2. Exploración de datos (EDA)...")
        explore_data(self.df.copy(), self.config.TARGET_CLASSIFICATION)
        return self

    def preprocess_data(self):
        """ Fase 3: Limpieza y preprocesamiento de nulos/tipos. """
        print("3. Preprocesamiento...")
        self.df = handle_missing_data(self.df)
        return self

    def feature_engineering(self):
        """ Fase 4: Creación de features (engineer.py). """
        print("4. Feature Engineering...")
        df_engineered = engineer_features(self.df.copy())

        # Split y Escalado
        X_train, X_test, y_train, y_test = split_data(df_engineered, self.config.TARGET_CLASSIFICATION)
        X_train_scaled, X_test_scaled, _ = apply_scaling(X_train, X_test)

        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test

        print(f"Datos listos. Train shape: {self.X_train.shape}")
        return self

    def train_evaluate(self):
        """ Fase 5: Entrenamiento del modelo. """
        print("5. Entrenamiento del modelo...")
        n_features = self.X_train.shape
        model = build_mlp_classifier(n_features)

        # Simulación de entrenamiento (usaría DataLoader o tf.data)
        # model.fit(...)
        # La evaluación rigurosa se haría en el conjunto de prueba (test)

        self.model = model # Guarda el modelo entrenado
        return self

    def run_full_pipeline(self):
        """ Orquesta todas las fases. """
        self.load_data().explore_data().preprocess_data().feature_engineering().train_evaluate()
        # La interpretabilidad se suele ejecutar en un script separado
```

### 3. Archivos de Ejecución (Scripts)

#### `scripts/run_pipeline.py`

El script principal de ejecución CLI (Command Line Interface):

```
import sys
from config.config import Config
from src.pipeline import MLPipeline

def main():
    """ Invocación principal del pipeline ML/DL. """

    try:
        config = Config()
        pipeline = MLPipeline(config)
        pipeline.run_full_pipeline()

        # Guardar artefactos clave
        # pipeline.model.save(config.MODEL_FILENAME)
        print("\n✅ Pipeline ejecutado con éxito. Modelo entrenado guardado.")

    except FileNotFoundError as e:
        print(f"Error fatal: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado durante la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### `scripts/step4_interpretability.py`

Script para generar las explicaciones post-modelo, utilizando métodos como SHAP o _Permutation Importance_ para datos tabulares:

```
import joblib
import pandas as pd
import shap # Librería post-hoc para interpretabilidad
from config.config import Config
from src.pipeline import MLPipeline

def run_interpretabilidad(X_test: pd.DataFrame, model_path: Path):
    """ Carga el modelo y genera las explicaciones SHAP. """

    # 1. Cargar modelo previamente entrenado
    try:
        model = joblib.load(model_path) # Usando joblib si es un modelo Scikit-learn
        # model = tf.keras.models.load_model(model_path) # Si es Keras
    except:
        print("Modelo no encontrado. Asegúrate de ejecutar el entrenamiento primero.")
        return

    # 2. Interpretabilidad con SHAP para datos tabulares
    # Se usaría el explainer adecuado (TreeExplainer para árboles, KernelExplainer para redes)
    print("\nGenerando explicaciones SHAP...")

    # Placeholder: Asumimos un modelo Scikit-learn entrenado para este ejemplo
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_test)

    # 3. Generar Summary Plot (artefacto)
    # shap.summary_plot(shap_values, X_test, show=False)
    # plt.savefig(Config.OUTPUTS_EDA_PATH / "shap_summary.png")

    # 4. Análisis de Importancia de Variables
    print("Top 5 factores que influyen en la predicción de Modalidad (ejemplo):")
    # Este análisis confirmaría si 'Área CINE' o 'POST_2020' son predictores clave
    # ...

if __name__ == "__main__":
    # Para el ejemplo, cargaríamos los datos de test ya procesados
    # X_test = pd.read_parquet(Config.PROCESSED_TEST_FILE)
    # run_interpretabilidad(X_test, Config.MODEL_FILENAME)
    pass
```

### 4. Archivos de Datos y Artefactos

#### `data/raw/data_input.csv` (Extracto del encabezado)

Muestra las variables fuente mencionadas en el proyecto:

```
ANIO,REGION,INSTITUCION,AREA_CINE,DURACION_TOTAL_SEMESTRES,PROMEDIO_EDAD_PROGRAMA,MODALIDAD_BIN,TITULACIONES_MUJERES,TITULACIONES_HOMBRES,...
2023,RM,UChile,INGENIERIA,10,25.5,Presencial,50,150,...
2022,VII,INACAP,ADMINISTRACION,4,38.1,No presencial,120,80,...
2023,VIII,UdeC,SALUD,12,24.0,Presencial,80,20,...
...
```

#### `outputs/eda/figures/distribucion_obj.png` (Descripción)

Este sería un **gráfico de barras** generado por `src/eda.py` que muestra los conteos y proporciones de la variable objetivo de clasificación (`MODALIDAD_BIN`). Si el conteo de "No presencial" fuera significativamente menor que "Presencial", el gráfico evidenciaría un **desbalance** (si _Imbalance Ratio_ $\ge 1.5$), lo que justificaría el uso de métricas como **AUC-PR** y **F1-macro**.

#### `models/trained/final_model.pkl` (Descripción)

Archivo binario serializado que contiene el modelo MLP entrenado, listo para ser cargado y utilizado para inferencia o interpretabilidad.

### 5. Documentación de Entorno
