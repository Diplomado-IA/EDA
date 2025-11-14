#!/usr/bin/env python3
"""
Script de verificación rápida del pipeline
Verifica que todas las fases estén correctamente implementadas
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MLPipeline
from src.config import Config

def check_phase_1():
    """Verificar Fase 1 (EDA)"""
    print("\n" + "="*70)
    print("✅ FASE 1: ANÁLISIS EXPLORATORIO (EDA)")
    print("="*70)
    
    eda_outputs = Path("outputs/eda")
    expected_files = [
        "01_values_count.png",
        "02_edad_distribucion.png",
        "03_distribution_program.png",
        "04_correlation_matrix.png",
        "05_missing_values.png",
        "06_outliers_detection.png"
    ]
    
    generated = 0
    for file in expected_files:
        path = eda_outputs / file
        status = "✅" if path.exists() else "❌"
        size = f"({path.stat().st_size / 1024:.0f} KB)" if path.exists() else ""
        print(f"{status} {file} {size}")
        if path.exists():
            generated += 1
    
    print(f"\n📊 Resultado: {generated}/{len(expected_files)} archivos generados (dataset grande activo)")
    return generated == len(expected_files)

def check_phase_2():
    """Verificar Fase 2 (Feature Engineering)"""
    print("\n" + "="*70)
    print("✅ FASE 2: FEATURE ENGINEERING")
    print("="*70)
    
    data_processed = Path("data/processed")
    expected_files = [
        "X_train_engineered.pkl",
        "X_test_engineered.pkl",
        "correlation_matrix.csv",
        "vif_scores.csv",
        "selected_features.txt",
        "feature_engineering_report.txt"
    ]
    
    generated = 0
    for file in expected_files:
        path = data_processed / file
        status = "✅" if path.exists() else "❌"
        size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if path.exists() and path.stat().st_size > 1024*1024 else \
               f"({path.stat().st_size / 1024:.0f} KB)" if path.exists() else ""
        print(f"{status} {file} {size}")
        if path.exists():
            generated += 1
    
    print(f"\n🎨 Resultado: {generated}/{len(expected_files)} archivos generados")
    return generated == len(expected_files)

def check_configuration():
    """Verificar configuración"""
    print("\n" + "="*70)
    print("⚙️ CONFIGURACIÓN DEL PROYECTO")
    print("="*70)
    
    config = Config()
    
    print(f"📊 Dataset: {config.DATASET_PATH}")
    print(f"📍 Existe: {'✅' if config.DATASET_PATH.exists() else '❌'}")
    
    if config.DATASET_PATH.exists():
        size_mb = config.DATASET_PATH.stat().st_size / 1024 / 1024
        print(f"💾 Tamaño: {size_mb:.1f} MB")
    
    print(f"\n🎯 Variables Objetivo:")
    print(f"  • Clasificación: {config.TARGET_CLASSIFICATION}")
    print(f"  • Regresión: {config.TARGET_REGRESSION}")
    
    print(f"\n📈 Configuración:")
    print(f"  • Train/Test Split: {config.TRAIN_TEST_SPLIT}")
    print(f"  • Random State: {config.RANDOM_STATE}")
    print(f"  • Encoding: {config.ENCODING}")
    
    return True

def main():
    """Función principal"""
    print("\n" + "█"*70)
    print("🚀 VERIFICACIÓN DEL PIPELINE - PROYECTO ML")
    print("█"*70)
    
    try:
        # Verificar configuración
        check_configuration()
        
        # Verificar Fase 1
        phase1_ok = check_phase_1()
        
        # Verificar Fase 2
        phase2_ok = check_phase_2()
        
        # Resumen
        print("\n" + "="*70)
        print("📋 RESUMEN DE VERIFICACIÓN")
        print("="*70)
        print(f"✅ Fase 1 (EDA): {'COMPLETADA' if phase1_ok else 'PENDIENTE'}")
        print(f"✅ Fase 2 (Feature Engineering): {'COMPLETADA' if phase2_ok else 'PENDIENTE'}")
        
        if phase1_ok and phase2_ok:
            print("\n" + "🎉"*35)
            print("✅ TODAS LAS FASES VERIFICADAS CORRECTAMENTE")
            print("🎉"*35)
            print("\n▶️ Puedes ejecutar:")
            print("   • python run_pipeline.py full")
            print("   • streamlit run ui/pipeline_executor.py")
        else:
            print("\n⚠️  Algunas fases falta generar. Ejecuta:")
            print("   • python run_pipeline.py full")
        
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error durante verificación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
