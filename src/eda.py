# src/eda.py
from pathlib import Path
from config.config import Config
import pandas as pd

ENCODINGS_TRY = ("utf-8", "latin-1", "cp1252")

def cargar_csv(ruta: str | Path, sep: str | None = None, encodings: tuple[str,...]=ENCODINGS_TRY) -> pd.DataFrame:
    """Carga robusta de CSV con prueba de múltiples encodings y validación de ruta.
    - Usa separador de Config si no se especifica.
    - low_memory=False para tipos consistentes.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    sep = sep or Config.SEPARATOR
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Fallo al leer CSV usando encodings {encodings}. Último error: {last_err}")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional

# Para resolver la referencia circular en eda_minimo
from .resumen_columnas import resumen_columnas, ordenar_resumen, obtener_top_faltantes


def mostrar_info_basica(df: pd.DataFrame, n_head: int = 5) -> None:
    print("\n=== Vista rápida del dataset ===")
    print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
    print("\nColumnas y tipos:")
    print(df.dtypes)
    print(f"\nPrimeras {n_head} filas:")
    print(df.head(n_head))


def plot_distrib_objetivo(
    df: pd.DataFrame, objetivo: str, show: bool = True, savepath: str | None = None
) -> float:
    """
    Grafica la distribución de la variable objetivo y devuelve el 'imbalance ratio'.
    """
    if objetivo not in df.columns:
        raise KeyError(f"La columna objetivo '{objetivo}' no existe en el dataset.")

    vc = df[objetivo].value_counts(dropna=False)
    prop = (vc / len(df)).round(4)

    print("\n=== Distribución de la variable objetivo ===")
    print("Conteo por clase:\n", vc)
    print("Proporción por clase:\n", prop)

    imbalance_ratio = float("inf")
    if vc.min() > 0:
        imbalance_ratio = float(vc.max() / vc.min())
    print(f"Imbalance ratio (mayor/menor): {imbalance_ratio:.2f}")

    ax = vc.plot(kind="bar")
    ax.set_title(f"Distribución de la variable objetivo: {objetivo}")
    ax.set_xlabel("Clases")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
        print(f"[OK] Gráfico guardado en: {savepath}")

    if show:
        plt.show()
    else:
        plt.close()

    if imbalance_ratio >= 1.5:
        print(
            "⚠️ Desbalance: usa 'stratify' en el split; reporta F1/AUC; evalúa 'class_weight' o re-muestreo."
        )
    return imbalance_ratio


def descriptivos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas para las columnas numéricas del DataFrame.

    Args:
        df: DataFrame a analizar

    Returns:
        DataFrame con estadísticas descriptivas transpuesto (filas=columnas originales)
    """
    # Filtrar solo las columnas numéricas
    df_num = df.select_dtypes(include=["number"])

    if df_num.empty:
        print("⚠️ No se encontraron columnas numéricas en el dataset")
        return pd.DataFrame()

    # Generar descriptivos y transponer para mejor visualización
    desc = df_num.describe().T

    desc["cv"] = desc["std"] / desc["mean"]  # Coeficiente de variación
    desc["iqr"] = desc["75%"] - desc["25%"]  # Rango intercuartil
    desc["skewness"] = df_num.apply(lambda x: x.skew())  # Asimetría
    desc["kurtosis"] = df_num.apply(lambda x: x.kurtosis())  # Curtosis

    desc = desc.round(4)

    return desc


def top_categorias(
    df: pd.DataFrame, columnas: Optional[List[str]] = None, k: int = 10
) -> Dict[str, pd.Series]:
    """
    Obtiene las top-k categorías más frecuentes para las columnas categóricas especificadas.

    Args:
        df: DataFrame a analizar
        columnas: Lista de columnas categóricas a analizar. Si es None, se seleccionan
                  automáticamente hasta 5 columnas categóricas representativas.
        k: Número de categorías principales a devolver (por defecto 10)

    Returns:
        Diccionario con columnas como claves y Series de frecuencias como valores
    """
    # Si no se especifican columnas, seleccionar automáticamente
    if columnas is None:
        # Obtener columnas categóricas
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if not cat_cols:
            print("⚠️ No se encontraron columnas categóricas en el dataset")
            return {}

        # Filtrar columnas con demasiados valores únicos (como URLs o nombres)
        # Seleccionamos columnas con menos de 100 valores únicos
        filtered_cols = [col for col in cat_cols if df[col].nunique() < 100]

        # Si no hay columnas filtradas, usar las originales
        if not filtered_cols:
            filtered_cols = cat_cols

        # Seleccionar hasta 5 columnas representativas (con menos valores únicos)
        representativas = sorted(filtered_cols, key=lambda x: df[x].nunique())[:5]
        columnas = representativas

        print(f"Columnas categóricas seleccionadas automáticamente: {columnas}")

    # Validar que todas las columnas existan
    for col in columnas:
        if col not in df.columns:
            raise KeyError(f"La columna '{col}' no existe en el dataset.")

    resultado = {}

    # Calcular top categorías para cada columna
    for col in columnas:
        value_counts = df[col].value_counts(dropna=False).head(k)
        resultado[col] = value_counts

    return resultado


def graficar_histogramas(
    df: pd.DataFrame, max_cols: int = 12, show: bool = False
) -> list[str]:
    """
    Genera histogramas para todas las columnas numéricas del DataFrame y los guarda en archivos.

    Args:
        df: DataFrame con las columnas a graficar
        max_cols: Número máximo de columnas a graficar (para prevenir sobrecarga)
        show: Si es True, muestra los gráficos en pantalla

    Returns:
        Lista con las rutas a los archivos generados
    """
    # Filtrar columnas numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        print("⚠️ No se encontraron columnas numéricas para generar histogramas")
        return []

    # Si hay demasiadas columnas, advertir y limitar
    if len(num_cols) > max_cols:
        print(
            f"⚠️ Limitando a {max_cols} histogramas (de {len(num_cols)} columnas numéricas)"
        )
        num_cols = num_cols[:max_cols]

    rutas_guardado = []

    # Generar histograma para cada columna
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Generar el histograma
        df[col].hist(ax=ax, bins=20, edgecolor="black", alpha=0.7)

        # Configurar el gráfico
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()

        # Guardar el archivo
        ruta = f"outputs/eda/figures/hist_{col}.png"
        plt.savefig(ruta, dpi=150)
        rutas_guardado.append(ruta)

        if show:
            plt.show()
        else:
            plt.close()

    print(f"[OK] {len(rutas_guardado)} histogramas guardados en outputs/figures/")
    return rutas_guardado


def graficar_boxplots(
    df: pd.DataFrame, max_cols: int = 12, show: bool = False
) -> list[str]:
    """
    Genera boxplots horizontales para todas las columnas numéricas del DataFrame y los guarda.

    Args:
        df: DataFrame con las columnas a graficar
        max_cols: Número máximo de columnas a graficar
        show: Si es True, muestra los gráficos en pantalla

    Returns:
        Lista con las rutas a los archivos generados
    """
    # Filtrar columnas numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        print("⚠️ No se encontraron columnas numéricas para generar boxplots")
        return []

    # Si hay demasiadas columnas, advertir y limitar
    if len(num_cols) > max_cols:
        print(
            f"⚠️ Limitando a {max_cols} boxplots (de {len(num_cols)} columnas numéricas)"
        )
        num_cols = num_cols[:max_cols]

    rutas_guardado = []

    # Generar boxplot para cada columna
    for col in num_cols:
        plt.figure(figsize=(10, 4))
        ax = plt.subplot()

        # Generar el boxplot horizontal
        df[[col]].boxplot(ax=ax, vert=False)

        # Configurar el gráfico
        ax.set_title(f"Boxplot de {col}")
        ax.set_xlabel(col)
        ax.set_yticks([])  # Ocultar etiquetas del eje Y
        plt.tight_layout()

        # Guardar el archivo
        ruta = f"outputs/eda/figures/box_{col}.png"
        plt.savefig(ruta, dpi=150)
        rutas_guardado.append(ruta)

        if show:
            plt.show()
        else:
            plt.close()

    print(f"[OK] {len(rutas_guardado)} boxplots guardados en outputs/figures/")
    return rutas_guardado


def decision_metrica(imbalance_ratio: float) -> str:
    """
    Decide la métrica de evaluación y estrategia de entrenamiento según el imbalance ratio.

    Args:
        imbalance_ratio: Relación entre clase mayoritaria y minoritaria

    Returns:
        Mensaje con la decisión de métrica y estrategia
    """
    if imbalance_ratio >= 1.5:
        return (
            "Datos desbalanceados (IR={:.2f}):\n"
            "- Métricas: F1, AUC-PR (Precision-Recall)\n"
            "- Estrategias: Usar stratify=y en train_test_split\n"
            "- Opciones: class_weight='balanced', técnicas de resampling (SMOTE, RandomUnderSampler)"
        ).format(imbalance_ratio)
    else:
        return (
            "Datos balanceados (IR={:.2f}):\n"
            "- Métricas base: Accuracy, ROC-AUC\n"
            "- No se requieren técnicas especiales de balanceo"
        ).format(imbalance_ratio)


def eda_minimo(
    df: pd.DataFrame,
    objetivo: Optional[str] = None,
    max_cats: int = 10,
    no_show: bool = True,
    no_histos: bool = False,
    no_box: bool = False,
) -> Dict[str, List[str]]:
    """
    Ejecuta un análisis exploratorio de datos completo y genera todos los entregables.

    Args:
        df: DataFrame a analizar
        objetivo: Nombre de la columna objetivo (opcional)
        max_cats: Número máximo de categorías a mostrar por columna categórica
        no_show: Si es True, no muestra gráficos en pantalla
        no_histos: Si es True, omite la generación de histogramas
        no_box: Si es True, omite la generación de boxplots

    Returns:
        Diccionario con las rutas de todos los archivos generados
    """
    resultados = {
        "resumen": [],
        "categorias": [],
        "graficos": [],
    }

    # 1. Info básica
    print("\n=== Ejecutando EDA mínimo ===")
    mostrar_info_basica(df)

    # 2. Resumen de columnas
    resumen = resumen_columnas(df)
    resumen_path = "outputs/eda/resumen/resumen_columnas.csv"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8")
    print(f"[OK] Resumen de columnas guardado en: {resumen_path}")
    resultados["resumen"].append(resumen_path)

    # 2.1 Resumen ordenado
    resumen_ordenado = ordenar_resumen(resumen)
    resumen_ordenado_path = "outputs/eda/resumen/resumen_columnas_ordenado.csv"
    resumen_ordenado.to_csv(resumen_ordenado_path, index=False, encoding="utf-8")
    print(f"[OK] Resumen ordenado guardado en: {resumen_ordenado_path}")
    resultados["resumen"].append(resumen_ordenado_path)

    # 2.2 Top faltantes
    top10_faltantes = obtener_top_faltantes(resumen, top=10)
    top10_faltantes_path = "outputs/eda/resumen/top10_faltantes.csv"
    top10_faltantes.to_csv(top10_faltantes_path, index=False, encoding="utf-8")
    print(
        f"[OK] Top-10 columnas con más valores faltantes guardado en: {top10_faltantes_path}"
    )
    resultados["resumen"].append(top10_faltantes_path)

    # 3. Descriptivos numéricos
    desc_num = descriptivos_numericos(df)
    if not desc_num.empty:
        Path("outputs/eda/resumen").mkdir(parents=True, exist_ok=True)
        desc_num_path = "outputs/eda/resumen/descriptivos_numericos.csv"
        desc_num.to_csv(desc_num_path, encoding="utf-8")
        print(f"[OK] Descriptivos numéricos guardados en: {desc_num_path}")
        resultados["resumen"].append(desc_num_path)

    # 4. Top categorías
    tops = top_categorias(df, columnas=None, k=max_cats)
    for col, serie in tops.items():
        df_top = serie.reset_index()
        df_top.columns = [col, "frecuencia"]
        top_path = f"outputs/eda/resumen/topcats_{col}.csv"
        df_top.to_csv(top_path, index=False, encoding="utf-8")
        print(f"[OK] Top {max_cats} categorías de '{col}' guardado en: {top_path}")
        resultados["categorias"].append(top_path)

    # 5. Distribución objetivo (si se especifica)
    imbalance_ratio = None
    if objetivo:
        if objetivo in df.columns:
            fig_path = "outputs/eda/figures/objetivo_barras.png"
            imbalance_ratio = plot_distrib_objetivo(
                df, objetivo, show=(not no_show), savepath=fig_path
            )
            resultados["graficos"].append(fig_path)

            # 5.1 Decisión de métrica según IR
            decision = decision_metrica(imbalance_ratio)
            decision_path = "outputs/eda/resumen/decision_metricas.txt"
            with open(decision_path, "w", encoding="utf-8") as f:
                f.write(decision)
            print(f"[OK] Decisión de métrica guardada en: {decision_path}")
            resultados["resumen"].append(decision_path)
        else:
            print(f"⚠️ La columna objetivo '{objetivo}' no existe en el dataset")

    # 6. Histogramas (si no se omiten)
    if not no_histos:
        hist_paths = graficar_histogramas(df, show=(not no_show))
        resultados["graficos"].extend(hist_paths)

    # 7. Boxplots (si no se omiten)
    if not no_box:
        box_paths = graficar_boxplots(df, show=(not no_show))
        resultados["graficos"].extend(box_paths)

    print("\n=== EDA mínimo completado ===")
    # Crear índice de artefactos
    index_path = "outputs/eda/resumen/eda_index.txt"
    with open(index_path, 'w', encoding='utf-8') as f:
        for k,v in resultados.items():
            for r in v:
                f.write(f"{k}\t{r}\n")
    resultados['resumen'].append(index_path)
    print(f"- Archivos de resumen: {len(resultados['resumen'])}")
    print(f"- Archivos de categorías: {len(resultados['categorias'])}")
    print(f"- Gráficos generados: {len(resultados['graficos'])}")

    return resultados
