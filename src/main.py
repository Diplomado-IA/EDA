# Script principal para ejecutar el módulo 1 (EDA)
import argparse
import os
from pathlib import Path

from .cargar_csv import cargar_csv
from .resumen_columnas import resumen_columnas
from .eda import mostrar_info_basica, plot_distrib_objetivo

def ensure_dirs():
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/resumen").mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Pipeline EDA rápido")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--sep", default=",", help="Separador (por defecto ',')")
    parser.add_argument("--objetivo", default=None, help="Nombre de la columna objetivo (opcional)")
    parser.add_argument("--no-show", action="store_true", help="No mostrar gráficos en pantalla")
    args = parser.parse_args()

    ensure_dirs()

    # 1) Carga
    df = cargar_csv(args.csv, sep=args.sep)

    # 2) Info básica
    mostrar_info_basica(df)

    # 3) Resumen de columnas
    resumen = resumen_columnas(df)
    resumen_path = "outputs/resumen/resumen_columnas.csv"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8")
    print(f"[OK] Resumen de columnas guardado en: {resumen_path}")

    # 4) Distribución objetivo (opcional)
    if args.objetivo:
        fig_path = "outputs/figures/objetivo_barras.png"
        plot_distrib_objetivo(df, args.objetivo, show=(not args.no_show), savepath=fig_path)

if __name__ == "__main__":
    main()
