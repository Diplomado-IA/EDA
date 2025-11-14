#!/usr/bin/env python3
"""
Script rápido para ejecutar el pipeline completo desde línea de comandos
Uso: python run_pipeline.py [eda|full]
"""

import sys
import logging
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MLPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Función principal"""
    
    # Determinar modo
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    print("\n" + "="*70)
    print("🚀 PIPELINE ML - EDUCACIÓN SUPERIOR")
    print("="*70 + "\n")
    
    try:
        pipeline = MLPipeline()
        
        if mode == "eda":
            print("📊 Ejecutando solo EDA...\n")
            report = pipeline.run_eda_only()
            print("\n✅ EDA completado exitosamente")
            
        elif mode == "full":
            print("📊 Ejecutando pipeline completo...\n")
            results = pipeline.run_full_pipeline()
            print("\n✅ Pipeline completado exitosamente")
            
        else:
            print(f"❌ Modo '{mode}' no reconocido")
            print("Uso: python run_pipeline.py [eda|full]")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("✅ EJECUCIÓN COMPLETADA")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
