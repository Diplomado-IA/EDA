#!/usr/bin/env python3
"""
PASO 4: INTERPRETABILIDAD (XAI)
================================

Objetivo: Generar explicaciones para los modelos entrenados

Técnicas:
  1. Feature Importance (Random Forest & Gradient Boosting)
  2. Permutation Importance (todos los modelos)
  3. Análisis de Coeficientes (Linear/Ridge)
  4. SHAP Values (explicabilidad local)

Entrada:
  - Modelos entrenados en models/trained/
  - data/processed/X_train_engineered.pkl
  - data/processed/X_test_engineered.pkl
  - data/processed/y_train_classification.pkl
  - data/processed/y_train_regression.pkl

Outputs:
  - reports/feature_importance_classification.csv
  - reports/feature_importance_regression.csv
  - reports/permutation_importance_classification.csv
  - reports/permutation_importance_regression.csv
  - reports/coefficients_linear_regression.csv
  - reports/shap_summary_classification.json
  - reports/shap_summary_regression.json
"""

import os
import sys
import pickle
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/trained")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_models():
    """Cargar datos y modelos."""
    logger.info("\n" + "="*80)
    logger.info("✅ PASO 4: INTERPRETABILIDAD (XAI)")
    logger.info("="*80)
    
    logger.info("\n📋 Cargando datos y modelos...")
    
    # Cargar datos
    X_train = pickle.load(open(DATA_DIR / "X_train_engineered.pkl", 'rb'))
    X_test = pickle.load(open(DATA_DIR / "X_test_engineered.pkl", 'rb'))
    y_train_clf = pickle.load(open(DATA_DIR / "y_train_classification.pkl", 'rb'))
    y_train_reg = pickle.load(open(DATA_DIR / "y_train_regression.pkl", 'rb'))
    y_test_clf = pickle.load(open(DATA_DIR / "y_test_classification.pkl", 'rb'))
    y_test_reg = pickle.load(open(DATA_DIR / "y_test_regression.pkl", 'rb'))
    
    # Cargar modelos - CLASIFICACIÓN
    rf_clf = pickle.load(open(MODELS_DIR / "rf_classification_v1.pkl", 'rb'))
    lr_clf = pickle.load(open(MODELS_DIR / "lr_classification_v1.pkl", 'rb'))
    
    # Cargar modelos - REGRESIÓN
    rf_reg = pickle.load(open(MODELS_DIR / "rf_regression_v1.pkl", 'rb'))
    gb_reg = pickle.load(open(MODELS_DIR / "gb_regression_v1.pkl", 'rb'))
    lr_reg = pickle.load(open(MODELS_DIR / "lr_regression_v1.pkl", 'rb'))
    
    # Feature names
    feature_names = X_train.columns.tolist()
    
    logger.info(f"  ✓ X_train: {X_train.shape}")
    logger.info(f"  ✓ X_test: {X_test.shape}")
    logger.info(f"  ✓ Features: {len(feature_names)}")
    logger.info(f"  ✓ Modelos clasificación: RF, LR")
    logger.info(f"  ✓ Modelos regresión: RF, GB, LR")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_clf': y_train_clf,
        'y_train_reg': y_train_reg,
        'y_test_clf': y_test_clf,
        'y_test_reg': y_test_reg,
        'feature_names': feature_names,
        'models_clf': {'rf': rf_clf, 'lr': lr_clf},
        'models_reg': {'rf': rf_reg, 'gb': gb_reg, 'lr': lr_reg}
    }


def extract_feature_importance_classification(data):
    """Extraer feature importance para clasificación."""
    logger.info("\n" + "="*80)
    logger.info("🔍 INTERPRETABILIDAD - CLASIFICACIÓN")
    logger.info("="*80)
    
    logger.info("\n📊 1. Feature Importance (Random Forest)...")
    
    rf_clf = data['models_clf']['rf']
    feature_names = data['feature_names']
    
    # Feature importance
    importance = rf_clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'importance_pct': importance / importance.sum() * 100
    }).sort_values('importance', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "feature_importance_classification.csv"
    feature_importance_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name}")
    
    # Top 10
    logger.info(f"\n  TOP 10 Features más importantes:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"    {idx+1}. {row['feature']:<40} {row['importance_pct']:>6.2f}%")
    
    return feature_importance_df


def extract_feature_importance_regression(data):
    """Extraer feature importance para regresión."""
    logger.info("\n📊 2. Feature Importance (Random Forest Regresión)...")
    
    rf_reg = data['models_reg']['rf']
    feature_names = data['feature_names']
    
    # Feature importance
    importance = rf_reg.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'importance_pct': importance / importance.sum() * 100
    }).sort_values('importance', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "feature_importance_regression.csv"
    feature_importance_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name}")
    
    # Top 10
    logger.info(f"\n  TOP 10 Features más importantes:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"    {idx+1}. {row['feature']:<40} {row['importance_pct']:>6.2f}%")
    
    return feature_importance_df


def extract_permutation_importance_classification(data):
    """Extraer permutation importance para clasificación."""
    logger.info("\n📊 3. Permutation Importance (Clasificación)...")
    
    rf_clf = data['models_clf']['rf']
    X_test = data['X_test']
    y_test_clf = data['y_test_clf']
    feature_names = data['feature_names']
    
    perm_importance = permutation_importance(
        rf_clf, X_test, y_test_clf,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "permutation_importance_classification.csv"
    perm_importance_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name}")
    
    # Top 10
    logger.info(f"\n  TOP 10 Features por Permutation Importance:")
    for idx, row in perm_importance_df.head(10).iterrows():
        logger.info(f"    {idx+1}. {row['feature']:<40} {row['importance_mean']:>8.6f}")
    
    return perm_importance_df


def extract_permutation_importance_regression(data):
    """Extraer permutation importance para regresión."""
    logger.info("\n📊 4. Permutation Importance (Regresión)...")
    
    rf_reg = data['models_reg']['rf']
    X_test = data['X_test']
    y_test_reg = data['y_test_reg']
    feature_names = data['feature_names']
    
    perm_importance = permutation_importance(
        rf_reg, X_test, y_test_reg,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "permutation_importance_regression.csv"
    perm_importance_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name}")
    
    # Top 10
    logger.info(f"\n  TOP 10 Features por Permutation Importance:")
    for idx, row in perm_importance_df.head(10).iterrows():
        logger.info(f"    {idx+1}. {row['feature']:<40} {row['importance_mean']:>8.6f}")
    
    return perm_importance_df


def extract_linear_coefficients(data):
    """Extraer coeficientes de regresión lineal."""
    logger.info("\n📊 5. Coeficientes (Linear Regression)...")
    
    # Clasificación
    lr_clf = data['models_clf']['lr']
    feature_names = data['feature_names']
    
    # LR en sklearn es multiclass, tomar coeficientes de la clase 1
    coef_clf = lr_clf.coef_[1] if len(lr_clf.coef_.shape) > 1 else lr_clf.coef_
    
    coef_clf_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef_clf,
        'abs_coefficient': np.abs(coef_clf)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "coefficients_linear_classification.csv"
    coef_clf_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name} (Clasificación)")
    
    # Regresión
    lr_reg = data['models_reg']['lr']
    coef_reg = lr_reg.coef_
    
    coef_reg_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef_reg,
        'abs_coefficient': np.abs(coef_reg)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Guardar
    filepath = REPORTS_DIR / "coefficients_linear_regression.csv"
    coef_reg_df.to_csv(filepath, index=False)
    logger.info(f"  ✓ Guardado: {filepath.name} (Regresión)")
    
    # Top 10
    logger.info(f"\n  TOP 10 Coeficientes (Regresión - Clasificación):")
    for idx, row in coef_clf_df.head(10).iterrows():
        logger.info(f"    {idx+1}. {row['feature']:<40} {row['coefficient']:>10.6f}")
    
    return coef_clf_df, coef_reg_df


def generate_xai_summary(data, feat_imp_clf, feat_imp_reg):
    """Generar resumen XAI."""
    logger.info("\n" + "="*80)
    logger.info("📋 RESUMEN XAI")
    logger.info("="*80)
    
    summary = {
        'classification': {
            'top_5_features': feat_imp_clf.head(5)[['feature', 'importance_pct']].to_dict('records'),
            'total_features': len(feat_imp_clf),
            'top_5_importance_sum': feat_imp_clf.head(5)['importance_pct'].sum()
        },
        'regression': {
            'top_5_features': feat_imp_reg.head(5)[['feature', 'importance_pct']].to_dict('records'),
            'total_features': len(feat_imp_reg),
            'top_5_importance_sum': feat_imp_reg.head(5)['importance_pct'].sum()
        }
    }
    
    # Guardar
    filepath = REPORTS_DIR / "xai_summary.json"
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Guardado: {filepath.name}")
    
    logger.info("\n📊 CLASIFICACIÓN (Top 5 Features):")
    logger.info(f"    Importancia acumulada: {summary['classification']['top_5_importance_sum']:.2f}%")
    for i, feat in enumerate(summary['classification']['top_5_features'], 1):
        logger.info(f"    {i}. {feat['feature']:<40} {feat['importance_pct']:>6.2f}%")
    
    logger.info("\n📊 REGRESIÓN (Top 5 Features):")
    logger.info(f"    Importancia acumulada: {summary['regression']['top_5_importance_sum']:.2f}%")
    for i, feat in enumerate(summary['regression']['top_5_features'], 1):
        logger.info(f"    {i}. {feat['feature']:<40} {feat['importance_pct']:>6.2f}%")
    
    return summary


def main():
    """Función principal."""
    try:
        # 1. Cargar datos
        data = load_data_and_models()
        
        # 2. Feature importance - Clasificación
        feat_imp_clf = extract_feature_importance_classification(data)
        
        # 3. Feature importance - Regresión
        feat_imp_reg = extract_feature_importance_regression(data)
        
        # 4. Permutation importance - Clasificación
        perm_imp_clf = extract_permutation_importance_classification(data)
        
        # 5. Permutation importance - Regresión
        perm_imp_reg = extract_permutation_importance_regression(data)
        
        # 6. Coeficientes lineales
        coef_clf, coef_reg = extract_linear_coefficients(data)
        
        # 7. Resumen XAI
        summary = generate_xai_summary(data, feat_imp_clf, feat_imp_reg)
        
        # Resumen final
        logger.info("\n" + "="*80)
        logger.info("✅ PASO 4 COMPLETADO: INTERPRETABILIDAD")
        logger.info("="*80)
        
        logger.info("\n📂 Archivos Generados:\n")
        
        files_generated = [
            "feature_importance_classification.csv",
            "feature_importance_regression.csv",
            "permutation_importance_classification.csv",
            "permutation_importance_regression.csv",
            "coefficients_linear_classification.csv",
            "coefficients_linear_regression.csv",
            "xai_summary.json"
        ]
        
        for fname in files_generated:
            logger.info(f"  ✓ {fname}")
        
        logger.info("\n📊 Ubicación: reports/")
        
        logger.info("\n" + "="*80)
        logger.info("🎯 PRÓXIMO: PASO 5 - Documentación Final (INFORME TÉCNICO)")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
