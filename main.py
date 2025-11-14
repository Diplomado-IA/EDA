#!/usr/bin/env python3
"""Interfaz CLI principal del proyecto"""
import argparse
import logging
from pathlib import Path
import json

from src.config import Config
from src.pipeline import MLPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="🎓 Pipeline ML - Modelado Predictivo Educación Superior Chile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --mode eda
  python main.py --mode train
  python main.py --mode full
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        '--mode',
        required=True,
        choices=['eda', 'train', 'evaluate', 'full', 'config'],
        help='Modo de ejecución'
    )
    
    parser.add_argument(
        '--input',
        default=None,
        help='Ruta al dataset (default: config.DATASET_PATH)'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Directorio de salida (default: outputs/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verboso (DEBUG)'
    )
    
    args = parser.parse_args()
    
    # Ajustar logging si es verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Modo DEBUG activado")
    
    # Crear pipeline
    config = Config()
    
    # Actualizar paths si se proporcionan
    if args.input:
        config.DATASET_PATH = Path(args.input)
        logger.info(f"Dataset: {config.DATASET_PATH}")
    
    if args.output:
        config.OUTPUTS_DIR = Path(args.output)
        logger.info(f"Output: {config.OUTPUTS_DIR}")
    
    # Ejecutar según modo
    if args.mode == 'config':
        print("\n=== CONFIGURACIÓN DEL PROYECTO ===\n")
        config_dict = config.to_dict()
        print(json.dumps(config_dict, indent=2, default=str))
        
    elif args.mode == 'eda':
        logger.info("🔍 Iniciando Análisis Exploratorio de Datos (EDA)...")
        pipeline = MLPipeline(config)
        report = pipeline.run_eda_only()
        
        logger.info("✓ EDA completado")
        logger.info(f"Resultados guardados en: {config.OUTPUTS_DIR}/eda/")
        
    elif args.mode == 'train':
        logger.info("🚀 Iniciando entrenamiento de modelos...")
        pipeline = MLPipeline(config)
        pipeline.load_data()
        pipeline.explore_data()
        pipeline.preprocess_data()
        pipeline.train_models()
        
        logger.info("✓ Entrenamiento completado")
        
    elif args.mode == 'evaluate':
        logger.info("📊 Evaluando modelos...")
        pipeline = MLPipeline(config)
        pipeline.load_data()
        pipeline.explore_data()
        pipeline.preprocess_data()
        results = pipeline.evaluate_models()
        
        logger.info("✓ Evaluación completada")
        
    elif args.mode == 'full':
        logger.info("▶️  Ejecutando pipeline completo...")
        pipeline = MLPipeline(config)
        results = pipeline.run_full_pipeline()
        
        logger.info("\n" + "="*60)
        logger.info("✓✓✓ PIPELINE COMPLETADO ✓✓✓")
        logger.info("="*60)


if __name__ == '__main__':
    main()
