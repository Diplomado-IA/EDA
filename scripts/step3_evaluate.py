#!/usr/bin/env python3
"""
PASO 3: ENTRENAMIENTO DE REGRESIÓN
==================================

Objetivo: Entrenar modelos de regresión para predecir PROMEDIO EDAD PROGRAMA

Modelos:
  1. Linear Regression (baseline)
  2. Ridge Regression (L2 regularization)
  3. Random Forest Regressor
  4. Gradient Boosting Regressor

Entrada:
  - data/processed/X_train_engineered.pkl
  - data/processed/X_test_engineered.pkl
  - data/processed/y_train_regression.pkl
  - data/processed/y_test_regression.pkl

Outputs:
  - models/trained/lr_regression_v1.pkl
  - models/trained/ridge_regression_v1.pkl
  - models/trained/rf_regression_v1.pkl
  - models/trained/gb_regression_v1.pkl
  - models/metadata/regression_metrics.json
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/trained")
METADATA_DIR = Path("models/metadata")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Cargar datos de entrenamiento y prueba."""
    logger.info("\n" + "="*80)
    logger.info("✅ PASO 3: ENTRENAR MODELOS DE REGRESIÓN")
    logger.info("="*80)
    
    logger.info("\n📋 Verificando archivos requeridos...")
    
    files = {
        'X_train': DATA_DIR / "X_train_engineered.pkl",
        'X_test': DATA_DIR / "X_test_engineered.pkl",
        'y_train': DATA_DIR / "y_train_regression.pkl",
        'y_test': DATA_DIR / "y_test_regression.pkl"
    }
    
    for name, path in files.items():
        if path.exists():
            logger.info(f"  ✓ {path.name}")
        else:
            logger.error(f"  ✗ {path.name} NO ENCONTRADO")
            sys.exit(1)
    
    logger.info("\n✅ Todos los archivos requeridos encontrados")
    
    logger.info("\n📂 Cargando datos...")
    X_train = pickle.load(open(files['X_train'], 'rb'))
    X_test = pickle.load(open(files['X_test'], 'rb'))
    y_train = pickle.load(open(files['y_train'], 'rb'))
    y_test = pickle.load(open(files['y_test'], 'rb'))
    
    logger.info(f"  ✓ X_train: {X_train.shape}")
    logger.info(f"  ✓ X_test: {X_test.shape}")
    logger.info(f"  ✓ y_train: {y_train.shape}")
    logger.info(f"  ✓ y_test: {y_test.shape}")
    
    # Estadísticas de y_train
    logger.info(f"\n📊 Estadísticas TARGET (y_train):")
    logger.info(f"  • Media: {y_train.mean():.2f}")
    logger.info(f"  • Std: {y_train.std():.2f}")
    logger.info(f"  • Min: {y_train.min():.2f}")
    logger.info(f"  • Max: {y_train.max():.2f}")
    logger.info(f"  • Q1: {y_train.quantile(0.25):.2f}")
    logger.info(f"  • Mediana: {y_train.median():.2f}")
    logger.info(f"  • Q3: {y_train.quantile(0.75):.2f}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Entrenar modelos de regresión."""
    logger.info("\n" + "="*80)
    logger.info("🤖 FASE 3 - PASO 3: ENTRENAR REGRESIÓN (EDAD PROMEDIO)")
    logger.info("="*80)
    
    models = {}
    
    # 1. Linear Regression
    logger.info("\n📊 Modelo 1: Linear Regression (Baseline)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['lr'] = lr
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    logger.info(f"  ✓ Linear Regression entrenado")
    logger.info(f"    - Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    logger.info(f"    - Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # 2. Ridge Regression
    logger.info("\n📊 Modelo 2: Ridge Regression (L2 Regularization)...")
    logger.info("  • Hiperparámetros:")
    logger.info("    - alpha=1.0")
    logger.info("    - solver='auto'")
    ridge = Ridge(alpha=1.0, solver='auto')
    ridge.fit(X_train, y_train)
    models['ridge'] = ridge
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)
    logger.info(f"  ✓ Ridge Regression entrenado")
    logger.info(f"    - Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    logger.info(f"    - Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # 3. Random Forest Regressor
    logger.info("\n📊 Modelo 3: Random Forest Regressor...")
    logger.info("  • Hiperparámetros:")
    logger.info("    - n_estimators=100")
    logger.info("    - max_depth=20")
    logger.info("    - min_samples_split=10")
    logger.info("    - min_samples_leaf=5")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    logger.info(f"  ✓ Random Forest entrenado")
    logger.info(f"    - Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    logger.info(f"    - Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # 4. Gradient Boosting Regressor
    logger.info("\n📊 Modelo 4: Gradient Boosting Regressor...")
    logger.info("  • Hiperparámetros:")
    logger.info("    - n_estimators=100")
    logger.info("    - learning_rate=0.1")
    logger.info("    - max_depth=5")
    logger.info("    - min_samples_split=10")
    logger.info("    - min_samples_leaf=5")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['gb'] = gb
    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)
    logger.info(f"  ✓ Gradient Boosting entrenado")
    logger.info(f"    - Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    logger.info(f"    - Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Regresión: Entrenamiento completado")
    logger.info("="*80)
    
    return models


def cross_validate_regression(X_train, y_train, models, k=5):
    """Validación cruzada para regresión."""
    logger.info(f"\n⏱️  Realizando validación cruzada ({k}-Fold)...\n")
    
    cv_results = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }
    
    logger.info(f"🔄 Validación Cruzada (k={k}) - Regresión...")
    
    for model_name, model in models.items():
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        cv_results[model_name] = {
            'mae': {
                'mean': -cv_scores['test_mae'].mean(),
                'std': cv_scores['test_mae'].std()
            },
            'rmse': {
                'mean': -cv_scores['test_rmse'].mean(),
                'std': cv_scores['test_rmse'].std()
            },
            'r2': {
                'mean': cv_scores['test_r2'].mean(),
                'std': cv_scores['test_r2'].std()
            }
        }
        
        logger.info(f"\n  {model_name.upper()}:")
        logger.info(f"    MAE:  {cv_results[model_name]['mae']['mean']:.4f} ± {cv_results[model_name]['mae']['std']:.4f}")
        logger.info(f"    RMSE: {cv_results[model_name]['rmse']['mean']:.4f} ± {cv_results[model_name]['rmse']['std']:.4f}")
        logger.info(f"    R²:   {cv_results[model_name]['r2']['mean']:.4f} ± {cv_results[model_name]['r2']['std']:.4f}")
    
    return cv_results


def evaluate_models(X_test, y_test, models):
    """Evaluar modelos en test set."""
    logger.info("\n⏱️  Evaluando modelos en test set...\n")
    
    logger.info("="*80)
    logger.info("📊 EVALUACIÓN - REGRESIÓN (Test Set)")
    logger.info("="*80)
    
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"\n🔍 Modelo: {model_name.upper()}")
        logger.info(f"  • MAE:  {mae:.4f}")
        logger.info(f"  • MSE:  {mse:.4f}")
        logger.info(f"  • RMSE: {rmse:.4f}")
        logger.info(f"  • R²:   {r2:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Evaluación Regresión: Completada")
    logger.info("="*80)
    
    return results


def save_models(models):
    """Guardar modelos entrenados."""
    logger.info("\n💾 Guardando modelos...\n")
    
    logger.info(f"💾 Guardando modelos en {MODELS_DIR}/...")
    
    saved_models = []
    for model_name, model in models.items():
        filename = f"{model_name}_regression_v1.pkl"
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  ✓ {filename}")
        saved_models.append(filename)
    
    logger.info(f"\n✅ {len(saved_models)} modelos guardados")
    
    return saved_models


def save_results(cv_results, eval_results):
    """Guardar resultados en JSON."""
    logger.info("\n💾 Guardando resultados...\n")
    
    import json
    
    results_data = {
        'cross_validation': {},
        'test_evaluation': eval_results
    }
    
    # Convertir cv_results para serialización JSON
    for model_name, metrics in cv_results.items():
        results_data['cross_validation'][model_name] = {
            'mae': {
                'mean': float(metrics['mae']['mean']),
                'std': float(metrics['mae']['std'])
            },
            'rmse': {
                'mean': float(metrics['rmse']['mean']),
                'std': float(metrics['rmse']['std'])
            },
            'r2': {
                'mean': float(metrics['r2']['mean']),
                'std': float(metrics['r2']['std'])
            }
        }
    
    # Convertir eval_results para serialización JSON
    for model_name, metrics in results_data['test_evaluation'].items():
        for key in metrics:
            results_data['test_evaluation'][model_name][key] = float(metrics[key])
    
    filepath = METADATA_DIR / "regression_metrics.json"
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"💾 Guardando resultados en {METADATA_DIR}/...")
    logger.info(f"  ✓ regression_metrics.json")


def main():
    """Función principal."""
    try:
        # 1. Cargar datos
        X_train, X_test, y_train, y_test = load_data()
        
        # 2. Entrenar modelos
        models = train_models(X_train, X_test, y_train, y_test)
        
        # 3. Validación cruzada
        cv_results = cross_validate_regression(X_train, y_train, models, k=5)
        
        # 4. Evaluación en test set
        eval_results = evaluate_models(X_test, y_test, models)
        
        # 5. Guardar modelos
        save_models(models)
        
        # 6. Guardar resultados
        save_results(cv_results, eval_results)
        
        # 7. Resumen final
        logger.info("\n" + "="*80)
        logger.info("✅ PASO 3 COMPLETADO: REGRESIÓN")
        logger.info("="*80)
        
        logger.info("\n📊 RESULTADOS EN TEST SET:\n")
        logger.info("-"*80 + "\n")
        
        for model_name in sorted(models.keys()):
            logger.info(f"🔍 Modelo: {model_name.upper()}")
            logger.info(f"  • MAE:  {eval_results[model_name]['mae']:.4f}")
            logger.info(f"  • RMSE: {eval_results[model_name]['rmse']:.4f}")
            logger.info(f"  • R²:   {eval_results[model_name]['r2']:.4f}\n")
        
        logger.info("-"*80)
        
        logger.info("\n📂 Archivos Guardados:")
        for model_name in sorted(models.keys()):
            logger.info(f"  ✓ {model_name}_regression_v1.pkl")
        logger.info(f"  ✓ regression_metrics.json")
        
        logger.info("\n" + "="*80)
        logger.info("🎯 PRÓXIMO: PASO 4 - Interpretabilidad (XAI)")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
