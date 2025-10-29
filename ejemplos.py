# Función robusta para carga segura de csv
"""Ejemplo 1 — Carga robusta de CSV
Objetivo: Leer un CSV de forma segura, validando que el archivo exista 
y probando encodings comunes. Mostrar primeras filas 
y tipos de datos para verificar el dataset.
"""
print("\n"+"="*40)
print("Ejemplo 1 — Carga robusta de CSV\n")

import pandas as pd
from pathlib import Path

def cargar_csv(ruta:str, sep:str=",", encoding_prioridad=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Lee un archivo CSV probando múltiples 'encodings' y valida la existencia del archivo.

    Parámetros:
    - ruta: str. Ruta al CSV (relativa o absoluta).
    - sep: str. Separador de columnas ("," para CSV estándar, ";" típico de Excel en español).
    - encoding_prioridad: tupla de encodings a probar en orden.

    Retorna:
    - pd.DataFrame con los datos cargados.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta.resolve()}" )

    ultimo_error = None
    for enc in encoding_prioridad:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
            print(f"[OK] Cargado con encoding='{enc}' y separador='{sep}'. Filas={len(df)}, Columnas={df.shape[1]}" )
            return df
        except Exception as e:
            ultimo_error = e
            continue

    raise RuntimeError(f"No se pudo leer el CSV con encodings {encoding_prioridad}. Último error: {ultimo_error}")

# === USO DIDÁCTICO ===
df = cargar_csv("data/TITULADO_2007-2024_web_19_05_2025_E.csv", sep=";")
print(df.head(3))                 # Vista rápida de las primeras 3 filas
print(df.dtypes)                  # Tipos de cada columna



# Funciones para tablas de calidad y EDA
"""Ejemplo 2 — Tabla de calidad de columnas
Objetivo: Resumir por columna: tipo de dato, número y porcentaje de faltantes, cardinalidad
y una heurística que sugiere si una columna de texto podría ser binaria.
"""
print("\n"+"="*40)
print("Ejemplo 2 — Resumen de columnas\n")

import numpy as np
import pandas as pd  # (por si se ejecuta aislado)

def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla de 'calidad de datos' por columna:
    - dtype: tipo de dato
    - n_missing: número de faltantes
    - pct_missing: % de faltantes (0-100)
    - n_unique: cardinalidad (valores únicos)
    - maybe_binary_text: True/False si parece columna binaria guardada como texto
    """
    resumen = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean()*100).round(2),
        "n_unique": df.nunique(dropna=False)
    })

    def posible_binaria(col):
        vals = set(map(str, df[col].dropna().unique()))
        return vals.issubset({"0","1","True","False","true","false","Sí","No","si","no"})

    resumen["maybe_binary_text"] = [posible_binaria(c) for c in df.columns]
    return resumen.sort_values(["pct_missing","n_unique"], ascending=[False, True])

# === USO DIDÁCTICO ===
res = resumen_columnas(df)
print(res.head(10))


"""Ejemplo 3 — Estadísticos y top de categorías
Objetivo: Obtener descriptivos de columnas numéricas y,
para categóricas, listar categorías más frecuentes (top-k) para detectar valores
dominantes o basura.
"""
print("\n"+"="*40)
print("Ejemplo 3 — Estadísticos y top categorías\n")

import numpy as np

# 1) Descriptivos para columnas numéricas
desc_num = df.select_dtypes(include=[np.number]).describe().T
print(desc_num)   # Transpuesta para que cada fila sea una variable

# 2) Top de categorías para columnas no numéricas
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for c in cat_cols[:5]:  # Limita a 5 columnas para no saturar la salida
    print(f"\nTop 10 categorías en '{c}':")
    print(df[c].value_counts(dropna=False).head(10))


"""Ejemplo 4 — Distribución de objetivo + desbalance
Objetivo: Explorar la distribución de la variable objetivo para detectar
desbalance de clases y visualizarla con un gráfico de barras.
"""
print("\n"+"="*40)
print("Ejemplo 4 — Distribución de variable objetivo\n")

import matplotlib.pyplot as plt

objetivo = "company"   # <--- Cambia al nombre real de tu variable objetivo

if objetivo not in df.columns:
    raise KeyError(f"La columna objetivo '{objetivo}' no existe en el dataset.")

vc = df[objetivo].value_counts(dropna=False)
prop = (vc / len(df)).round(4)
print("Conteo por clase:\n", vc)
print("Proporción por clase:\n", prop)

imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
print(f"Imbalance ratio (mayor/menor): {imbalance_ratio:.2f}")

ax = vc.plot(kind="bar")
ax.set_title(f"Distribución de la variable objetivo: {objetivo}")
ax.set_xlabel("Clases"); ax.set_ylabel("Frecuencia")
plt.tight_layout(); plt.show()

if imbalance_ratio >= 1.5:
    print("⚠️ Desbalance: usa 'stratify' en el split; reporta F1/AUC; evalúa 'class_weight' o re-muestreo.")
