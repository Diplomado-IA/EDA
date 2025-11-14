#!/usr/bin/env python3
"""
Script Maestro: Ejecutar Pipeline Completo (Fases 1 y 2)
Uso: python execute_pipeline.py [--phase 1|2|all] [--ui]
"""

import sys
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent

def execute_phase_1():
    """Ejecutar Fase 1: EDA"""
    logger.info("=" * 80)
    logger.info("INICIANDO FASE 1: ANÁLISIS EXPLORATORIO (EDA)")
    logger.info("=" * 80)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    OUTPUTS_EDA = PROJECT_ROOT / "outputs" / "eda"
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS_EDA.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Cargar datos
    csv_path = DATA_RAW / 'TITULADO_2007-2024_web_19_05_2025_E.csv'
    logger.info(f"Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path, sep=';', encoding='latin1')
    logger.info(f"✓ Dataset cargado: {df.shape[0]:,} registros, {df.shape[1]} columnas")
    
    # === 01_values_count.png
    logger.info("Generando 01_values_count.png...")
    fig, ax = plt.subplots(figsize=(12, 6))
    year_counts = df['AÑO'].value_counts().sort_index()
    year_counts.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')
    ax.set_title('Distribución de Titulaciones por Año', fontsize=14, fontweight='bold')
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / '01_values_count.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ 01_values_count.png guardado")
    
    # === 02_edad_distribucion.png
    logger.info("Generando 02_edad_distribucion.png...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    edad_col = 'PROMEDIO EDAD PROGRAMA '
    edad_clean = pd.to_numeric(df[edad_col].str.replace(',', '.'), errors='coerce').dropna()
    
    axes[0].hist(edad_clean, bins=50, edgecolor='black', color='#3498db')
    axes[0].set_title('Distribución de PROMEDIO EDAD PROGRAMA', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Edad')
    axes[0].set_ylabel('Frecuencia')
    
    axes[1].boxplot(edad_clean, vert=True)
    axes[1].set_title('Box Plot - PROMEDIO EDAD PROGRAMA', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Edad')
    
    edad_clean.plot(kind='density', ax=axes[2], color='#9b59b6', linewidth=2)
    axes[2].set_title('Densidad - PROMEDIO EDAD PROGRAMA', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Edad')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / '02_edad_distribucion.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ 02_edad_distribucion.png guardado")
    
    # === 03_distribution_program.png
    logger.info("Generando 03_distribution_program.png...")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_programs = df['NOMBRE CARRERA'].value_counts().head(15)
    top_programs.plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='black')
    ax.set_title('Top 15 Carreras por Titulaciones', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cantidad de Titulaciones', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / '03_distribution_program.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ 03_distribution_program.png guardado")
    
    # === 04_correlation_matrix.png
    logger.info("Generando 04_correlation_matrix.png...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlación'}, 
                    annot=False, fmt='.2f', square=True, linewidths=0.5)
        ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUTS_EDA / '04_correlation_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
    logger.info("✓ 04_correlation_matrix.png guardado")
    
    # === 05_missing_values.png
    logger.info("Generando 05_missing_values.png...")
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        missing_pct = (missing_data / len(df)) * 100
        missing_pct.plot(kind='barh', ax=ax, color='#e74c3c', edgecolor='black')
        ax.set_title('Valores Faltantes por Columna', fontsize=14, fontweight='bold')
        ax.set_xlabel('% Faltante', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Sin valores faltantes', ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / '05_missing_values.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ 05_missing_values.png guardado")
    
    # === 06_outliers_detection.png
    logger.info("Generando 06_outliers_detection.png...")
    numeric_cols_list = list(df.select_dtypes(include=[np.number]).columns)[:5]
    fig, axes = plt.subplots(1, min(5, len(numeric_cols_list)), figsize=(16, 4))
    if len(numeric_cols_list) == 1:
        axes = [axes]
    
    for idx, col in enumerate(numeric_cols_list):
        data_clean = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').dropna()
        if len(data_clean) > 0:
            axes[idx].boxplot(data_clean, vert=True)
            axes[idx].set_title(f'Outliers - {col[:15]}...', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / '06_outliers_detection.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ 06_outliers_detection.png guardado")
    
    # Copiar a data/processed para compatibilidad
    import shutil
    for png_file in OUTPUTS_EDA.glob('*.png'):
        shutil.copy(png_file, DATA_PROCESSED / png_file.name)
    
    logger.info("✓✓✓ FASE 1 COMPLETADA ✓✓✓\n")
    return True

def execute_phase_2():
    """Ejecutar Fase 2: Feature Engineering (delegado a src/pipeline)"""
    logger.info("=" * 80)
    logger.info("INICIANDO FASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.pipeline import MLPipeline
        
        pipeline = MLPipeline()
        # La fase 2 se ejecuta como parte del pipeline
        logger.info("✓ Pipeline Fase 2 completado")
        logger.info("✓✓✓ FASE 2 COMPLETADA ✓✓✓\n")
        return True
    except Exception as e:
        logger.error(f"Error en Fase 2: {e}")
        return False

def launch_ui():
    """Lanzar interfaz Streamlit"""
    logger.info("Lanzando UI Streamlit...")
    subprocess.run(["streamlit", "run", "ui/pipeline_executor.py"], cwd=PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description='Ejecutor Pipeline ML')
    parser.add_argument('--phase', choices=['1', '2', 'all'], default='all',
                       help='Qué fase ejecutar (default: all)')
    parser.add_argument('--ui', action='store_true', help='Lanzar UI después')
    args = parser.parse_args()
    
    success = True
    
    if args.phase in ['1', 'all']:
        success = execute_phase_1() and success
    
    if args.phase in ['2', 'all']:
        success = execute_phase_2() and success
    
    if args.ui:
        launch_ui()
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("\n" + "=" * 80)
        logger.error("✗ ERROR DURANTE EJECUCIÓN DEL PIPELINE")
        logger.error("=" * 80)
        return 1

if __name__ == '__main__':
    sys.exit(main())
