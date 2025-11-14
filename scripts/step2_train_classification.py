#!/usr/bin/env python3
"""
PASO 2: ENTRENAR MODELOS DE CLASIFICACIÓN (MODALIDAD)
======================================================

Script para entrenar y evaluar modelos de clasificación en Fase 3.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("✅ PASO 2: ENTRENAR MODELOS DE CLASIFICACIÓN")
    logger.info("="*80)

    # Verificar que los datos de Fase 2 estén disponibles
    data_path = Path("data/processed")
    required_files = [
        "X_train_engineered.pkl",
        "X_test_engineered.pkl",
        "y_train_classification.pkl",
        "y_test_classification.pkl"
    ]

    logger.info("\n📋 Verificando archivos requeridos...")
    all_exist = True
    for file in required_files:
        filepath = data_path / file
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {file}")
        if not exists:
            all_exist = False

    if not all_exist:
        logger.error("\n❌ ERROR: Faltan archivos de Fase 2")
        logger.info("\nArchivos esperados en data/processed/:")
        for file in required_files:
            logger.info(f"  - {file}")
        return False

    logger.info("\n✅ Todos los archivos requeridos encontrados")

    # Cargar datos
    logger.info("\n📂 Cargando datos...")
    try:
        X_train = pd.read_pickle(data_path / "X_train_engineered.pkl")
        X_test = pd.read_pickle(data_path / "X_test_engineered.pkl")
        y_train_class = pd.read_pickle(data_path / "y_train_classification.pkl")
        y_test_class = pd.read_pickle(data_path / "y_test_classification.pkl")
        
        logger.info(f"  ✓ X_train: {X_train.shape}")
        logger.info(f"  ✓ X_test: {X_test.shape}")
        logger.info(f"  ✓ y_train: {y_train_class.shape}")
        logger.info(f"  ✓ y_test: {y_test_class.shape}")
    except Exception as e:
        logger.error(f"\n❌ Error cargando datos: {e}")
        return False

    # Verificar distribución de clases
    logger.info("\n📊 Distribución de clases (Train):")
    value_counts = y_train_class.value_counts()
    for clase, count in value_counts.items():
        pct = (count / len(y_train_class)) * 100
        logger.info(f"  {clase}: {count:,} ({pct:.1f}%)")

    logger.info("\n📊 Distribución de clases (Test):")
    value_counts = y_test_class.value_counts()
    for clase, count in value_counts.items():
        pct = (count / len(y_test_class)) * 100
        logger.info(f"  {clase}: {count:,} ({pct:.1f}%)")

    # Importar módulo de modelos
    logger.info("\n🔧 Importando módulo de modelos...")
    try:
        from src.models import ModelTrainer, ModelEvaluator
        logger.info("  ✓ ModelTrainer importado")
        logger.info("  ✓ ModelEvaluator importado")
    except Exception as e:
        logger.error(f"\n❌ Error importando módulo: {e}")
        return False

    # ===== ENTRENAR CLASIFICACIÓN =====
    logger.info("\n🤖 Inicializando entrenador...")
    trainer = ModelTrainer(random_state=42)

    logger.info("\n⏱️  Entrenando modelos de clasificación...")
    try:
        models_class = trainer.train_classification(X_train, y_train_class)
    except Exception as e:
        logger.error(f"\n❌ Error entrenando modelos: {e}")
        return False

    logger.info("\n⏱️  Realizando validación cruzada (5-Fold)...")
    try:
        cv_results_class = trainer.cross_validate_classification(X_train, y_train_class, k=5)
    except Exception as e:
        logger.error(f"\n❌ Error en validación cruzada: {e}")
        return False

    # ===== EVALUAR CLASIFICACIÓN =====
    logger.info("\n📊 Inicializando evaluador...")
    evaluator = ModelEvaluator()

    logger.info("\n⏱️  Evaluando modelos en test set...")
    try:
        results_class = evaluator.evaluate_classification(models_class, X_test, y_test_class)
    except Exception as e:
        logger.error(f"\n❌ Error evaluando modelos: {e}")
        return False

    # ===== GUARDAR MODELOS Y RESULTADOS =====
    logger.info("\n💾 Guardando modelos...")
    try:
        saved_models = trainer.save_models(output_dir="models/trained")
    except Exception as e:
        logger.error(f"\n❌ Error guardando modelos: {e}")
        return False

    logger.info("\n💾 Guardando resultados...")
    try:
        saved_results = evaluator.save_results(output_dir="models/metadata")
    except Exception as e:
        logger.error(f"\n❌ Error guardando resultados: {e}")
        return False

    # ===== RESUMEN FINAL =====
    logger.info("\n" + "="*80)
    logger.info("✅ PASO 2 COMPLETADO: CLASIFICACIÓN")
    logger.info("="*80)

    logger.info("\n📊 RESULTADOS EN TEST SET:")
    logger.info("\n" + "-"*80)
    for model_name, metrics in results_class.items():
        logger.info(f"\n🔍 Modelo: {model_name.upper()}")
        logger.info(f"  • Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  • Precision: {metrics['precision']:.4f}")
        logger.info(f"  • Recall:    {metrics['recall']:.4f}")
        logger.info(f"  • F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            logger.info(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")

    logger.info("\n" + "-"*80)
    logger.info("\n📂 Archivos Guardados:")
    for key, path in saved_models.items():
        logger.info(f"  ✓ {Path(path).name}")
    for key, path in saved_results.items():
        logger.info(f"  ✓ {Path(path).name}")

    logger.info("\n" + "="*80)
    logger.info("🎯 PRÓXIMO: PASO 3 - Entrenar Regresión")
    logger.info("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
