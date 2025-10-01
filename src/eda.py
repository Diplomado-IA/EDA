# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt

def mostrar_info_basica(df: pd.DataFrame, n_head: int = 5) -> None:
    print("\n=== Vista rápida del dataset ===")
    print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
    print("\nColumnas y tipos:")
    print(df.dtypes)
    print(f"\nPrimeras {n_head} filas:")
    print(df.head(n_head))

def plot_distrib_objetivo(df: pd.DataFrame, objetivo: str, show: bool = True, savepath: str | None = None) -> float:
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
        print("⚠️ Desbalance: usa 'stratify' en el split; reporta F1/AUC; evalúa 'class_weight' o re-muestreo.")
    return imbalance_ratio
