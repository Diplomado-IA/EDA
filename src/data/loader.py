"""
Módulo de carga y validación de datos.

Fase 0: Entendimiento del problema y datos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings


class DataLoader:
    """Cargador de datos con validación y metadatos."""
    
    def __init__(self, encoding_priority=("utf-8", "latin-1", "iso-8859-1")):
        """
        Inicializa el cargador de datos.
        
        Args:
            encoding_priority: Tuple de encodings a probar en orden
        """
        self.encoding_priority = encoding_priority
        self.metadata = {}
        
    def load_csv(
        self, 
        filepath: str, 
        sep: str = ",",
        low_memory: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Carga CSV con detección automática de encoding.
        
        Args:
            filepath: Ruta al archivo CSV
            sep: Separador de columnas
            low_memory: Si usar memoria baja
            
        Returns:
            Tuple (DataFrame, metadata_dict)
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            RuntimeError: Si no se puede leer con ningún encoding
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró: {filepath.resolve()}")
            
        ultimo_error = None
        for enc in self.encoding_priority:
            try:
                df = pd.read_csv(
                    filepath, 
                    sep=sep, 
                    encoding=enc, 
                    low_memory=low_memory
                )
                
                # Guardar metadata
                self.metadata = {
                    "nombre_archivo": filepath.name,
                    "ruta_absoluta": str(filepath.resolve()),
                    "separador": sep,
                    "encoding_usado": enc,
                    "filas": len(df),
                    "columnas": df.shape[1],
                    "tipos_columnas": df.dtypes.to_dict(),
                    "memoria_mb": df.memory_usage(deep=True).sum() / 1024**2,
                }
                
                print(f"[OK] Cargado con encoding='{enc}', "
                      f"filas={len(df)}, columnas={df.shape[1]}")
                return df, self.metadata
                
            except Exception as e:
                print(f"[WARN] Falló con encoding='{enc}': {e}")
                ultimo_error = e
                continue
                
        raise RuntimeError(
            f"No se pudo leer con encodings {self.encoding_priority}. "
            f"Último error: {ultimo_error}"
        )
    
    def validate_schema(
        self, 
        df: pd.DataFrame, 
        expected_columns: Optional[list] = None
    ) -> Dict:
        """
        Valida el esquema del DataFrame.
        
        Args:
            df: DataFrame a validar
            expected_columns: Lista de columnas esperadas (opcional)
            
        Returns:
            Dict con resultados de validación
        """
        validation = {
            "filas": len(df),
            "columnas": df.shape[1],
            "columnas_presentes": df.columns.tolist(),
            "duplicados": df.duplicated().sum(),
            "filas_vacias": df.isna().all(axis=1).sum(),
        }
        
        if expected_columns:
            missing = set(expected_columns) - set(df.columns)
            extra = set(df.columns) - set(expected_columns)
            validation["columnas_faltantes"] = list(missing)
            validation["columnas_extra"] = list(extra)
            validation["schema_valido"] = len(missing) == 0
        else:
            validation["schema_valido"] = True
            
        return validation
    
    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene resumen de calidad de datos.
        
        Args:
            df: DataFrame a resumir
            
        Returns:
            DataFrame con resumen por columna
        """
        # Calcular métricas básicas (excluir el índice de memory_usage)
        summary_data = {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=False),
            "memory_kb": (df.memory_usage(deep=True, index=False) / 1024).round(2),
        }
        
        summary = pd.DataFrame(summary_data)
        
        # Agregar ejemplos de valores
        sample_values = []
        for col in df.columns:
            try:
                samples = df[col].dropna().head(3).tolist()
                sample_values.append(str(samples[:3]))
            except Exception:
                sample_values.append("N/A")
        
        summary["sample_values"] = sample_values
        
        return summary.sort_values("pct_missing", ascending=False)


def load_titulados_data(filepath: str = "data/raw/TITULADO_2007-2024_web_19_05_2025_E.csv") -> Tuple[pd.DataFrame, Dict]:
    """
    Función de conveniencia para cargar datos de titulados.
    
    Args:
        filepath: Ruta al CSV de titulados
        
    Returns:
        Tuple (DataFrame, metadata)
    """
    loader = DataLoader()
    df, metadata = loader.load_csv(filepath, sep=";")
    
    # Validaciones específicas del dataset
    expected_cols = ["AÑO", "REGIÓN", "TOTAL TITULACIONES", "NOMBRE INSTITUCIÓN"]
    validation = loader.validate_schema(df, expected_cols)
    
    if not validation["schema_valido"]:
        warnings.warn(
            f"Columnas faltantes: {validation.get('columnas_faltantes', [])}"
        )
    
    print(f"\n[INFO] Dataset de titulados cargado correctamente")
    print(f"Período: {df['AÑO'].min()} - {df['AÑO'].max()}")
    print(f"Regiones: {df['REGIÓN'].nunique()}")
    print(f"Instituciones: {df['NOMBRE INSTITUCIÓN'].nunique()}")
    
    return df, metadata


if __name__ == "__main__":
    # Ejemplo de uso
    df, meta = load_titulados_data()
    loader = DataLoader()
    summary = loader.get_summary(df)
    print("\nResumen de calidad:")
    print(summary.head(10))
