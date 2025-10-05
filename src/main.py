# Script principal para ejecutar el módulo 1 (EDA)
import argparse
import os
from pathlib import Path

from .cargar_csv import cargar_csv
from .resumen_columnas import resumen_columnas, ordenar_resumen, obtener_top_faltantes
from .eda import (
    mostrar_info_basica,
    plot_distrib_objetivo,
    descriptivos_numericos,
    top_categorias,
    graficar_histogramas,
    graficar_boxplots,
    decision_metrica,
    eda_minimo,
)


def ensure_dirs():
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/resumen").mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline EDA rápido")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--sep", default=",", help="Separador (por defecto ',')")
    parser.add_argument(
        "--objetivo", default=None, help="Nombre de la columna objetivo (opcional)"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="No mostrar gráficos en pantalla"
    )
    parser.add_argument(
        "--cat-cols",
        nargs="+",
        default=None,
        help="Columnas categóricas para mostrar el top-10 (separadas por espacios)",
    )
    parser.add_argument(
        "--max-cats",
        type=int,
        default=10,
        help="Número máximo de categorías a mostrar por columna categórica (por defecto 10)",
    )
    parser.add_argument(
        "--no-histos",
        action="store_true",
        help="No generar histogramas para las variables numéricas",
    )
    parser.add_argument(
        "--no-box",
        action="store_true",
        help="No generar boxplots para las variables numéricas",
    )
    parser.add_argument(
        "--run-minimo",
        action="store_true",
        help="Ejecutar el EDA mínimo completo (todas las funciones en un solo paso)",
    )
    args = parser.parse_args()

    ensure_dirs()

    # 1) Carga
    df, metadata_carga = cargar_csv(args.csv, sep=args.sep)

    # 1.1) Persistir información de carga
    carga_info_path = "outputs/resumen/carga_info.txt"
    with open(carga_info_path, "w", encoding="utf-8") as f:
        f.write("=== INFORMACIÓN DE CARGA DEL DATASET ===\n\n")
        f.write(f"Nombre del archivo: {metadata_carga['nombre_archivo']}\n")
        f.write(f"Ruta absoluta: {metadata_carga['ruta_absoluta']}\n")
        f.write(f"Separador utilizado: '{metadata_carga['separador']}'\n")
        f.write(f"Encoding detectado: {metadata_carga['encoding_usado']}\n")
        f.write(
            f"Dimensiones: {metadata_carga['filas']} filas × {metadata_carga['columnas']} columnas\n\n"
        )
        f.write("Columnas y tipos:\n")
        for col, tipo in metadata_carga["tipos_columnas"].items():
            f.write(f"- {col}: {tipo}\n")
    print(f"[OK] Información de carga guardada en: {carga_info_path}")

    # 2) Info básica
    mostrar_info_basica(df)

    # 3) Resumen de columnas
    resumen = resumen_columnas(df)
    resumen_path = "outputs/resumen/resumen_columnas.csv"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8")
    print(f"[OK] Resumen de columnas guardado en: {resumen_path}")

    # 3.1) Resumen ordenado (%missing desc, n_unique asc)
    resumen_ordenado = ordenar_resumen(resumen)
    resumen_ordenado_path = "outputs/resumen/resumen_columnas_ordenado.csv"
    resumen_ordenado.to_csv(resumen_ordenado_path, index=False, encoding="utf-8")
    print(f"[OK] Resumen ordenado guardado en: {resumen_ordenado_path}")

    # 3.2) Top-10 columnas con más valores faltantes
    top10_faltantes = obtener_top_faltantes(resumen, top=10)
    top10_faltantes_path = "outputs/resumen/top10_faltantes.csv"
    top10_faltantes.to_csv(top10_faltantes_path, index=False, encoding="utf-8")
    print(
        f"[OK] Top-10 columnas con más valores faltantes guardado en: {top10_faltantes_path}"
    )

    # 3.3) Descriptivos numéricos
    desc_num = descriptivos_numericos(df)
    if not desc_num.empty:
        desc_num_path = "outputs/resumen/descriptivos_numericos.csv"
        desc_num.to_csv(desc_num_path, encoding="utf-8")
        print(f"[OK] Descriptivos numéricos guardados en: {desc_num_path}")

    # Si se pide el EDA mínimo, ejecutar todo en un solo paso
    if args.run_minimo:
        print("\nEjecutando EDA mínimo (todo en un solo paso)")
        eda_minimo(
            df,
            objetivo=args.objetivo,
            max_cats=args.max_cats,
            no_show=args.no_show,
            no_histos=args.no_histos,
            no_box=args.no_box,
        )
        return

    # 3.4) Top categorías por columna categórica
    tops = top_categorias(df, columnas=args.cat_cols, k=args.max_cats)
    for col, serie in tops.items():
        # Convertir a DataFrame para guardarlo como CSV
        df_top = serie.reset_index()
        df_top.columns = [col, "frecuencia"]

        # Guardar en CSV
        top_path = f"outputs/resumen/topcats_{col}.csv"
        df_top.to_csv(top_path, index=False, encoding="utf-8")
        print(f"[OK] Top {args.max_cats} categorías de '{col}' guardado en: {top_path}")

    # 4) Distribución objetivo (opcional) y decisión de métrica
    imbalance_ratio = None
    if args.objetivo:
        fig_path = "outputs/figures/objetivo_barras.png"
        imbalance_ratio = plot_distrib_objetivo(
            df, args.objetivo, show=(not args.no_show), savepath=fig_path
        )

        # 4.1) Guardar decisión de métrica según el desbalance
        decision = decision_metrica(imbalance_ratio)
        decision_path = "outputs/resumen/decision_metricas.txt"
        with open(decision_path, "w", encoding="utf-8") as f:
            f.write(decision)
        print(f"[OK] Decisión de métrica guardada en: {decision_path}")

    # 5) Histogramas (si no se omiten)
    if not args.no_histos:
        graficar_histogramas(df, show=(not args.no_show))

    # 6) Boxplots (si no se omiten)
    if not args.no_box:
        graficar_boxplots(df, show=(not args.no_show))


if __name__ == "__main__":
    main()
