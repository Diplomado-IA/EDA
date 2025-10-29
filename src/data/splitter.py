"""
Módulo de particionamiento de datos.

Fase 1: Datos y particiones
Evita data leakage con particiones temporales estrictas.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import json


class TemporalSplitter:
    """
    Divide datos por tiempo evitando data leakage.
    
    Siguiendo el caso de salmoneras:
    - Partición estricta por tiempo
    - Sin mezcla de entidades entre particiones del mismo período
    - Estratificación opcional
    """
    
    def __init__(self, time_col: str = "AÑO", entity_col: Optional[str] = None):
        """
        Args:
            time_col: Columna temporal para ordenar
            entity_col: Columna de entidad (ej: institución, región)
        """
        self.time_col = time_col
        self.entity_col = entity_col
        self.split_info = {}
        
    def split_by_years(
        self,
        df: pd.DataFrame,
        train_years: list,
        val_years: list,
        test_years: list,
        stratify_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide por años específicos.
        
        Args:
            df: DataFrame completo
            train_years: Lista de años para entrenamiento
            val_years: Lista de años para validación
            test_years: Lista de años para test
            stratify_col: Columna para estratificar (mantener proporciones)
            
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        # Extraer año numérico
        df = df.copy()
        df['_year_num'] = df[self.time_col].str.extract(r'(\d{4})').astype(int)
        
        # Filtrar por años
        train_df = df[df['_year_num'].isin(train_years)].copy()
        val_df = df[df['_year_num'].isin(val_years)].copy()
        test_df = df[df['_year_num'].isin(test_years)].copy()
        
        # Guardar info
        self.split_info = {
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_pct": len(train_df) / len(df) * 100,
            "val_pct": len(val_df) / len(df) * 100,
            "test_pct": len(test_df) / len(df) * 100,
        }
        
        # Validar sin traslape
        assert len(set(train_years) & set(val_years)) == 0, "Train y Val tienen años en común"
        assert len(set(train_years) & set(test_years)) == 0, "Train y Test tienen años en común"
        assert len(set(val_years) & set(test_years)) == 0, "Val y Test tienen años en común"
        
        # Limpiar columna temporal
        train_df = train_df.drop('_year_num', axis=1)
        val_df = val_df.drop('_year_num', axis=1)
        test_df = test_df.drop('_year_num', axis=1)
        
        print(f"[OK] Partición temporal completada:")
        print(f"  Train: {len(train_df):,} filas ({self.split_info['train_pct']:.1f}%)")
        print(f"  Val:   {len(val_df):,} filas ({self.split_info['val_pct']:.1f}%)")
        print(f"  Test:  {len(test_df):,} filas ({self.split_info['test_pct']:.1f}%)")
        
        if stratify_col:
            self._validate_stratification(df, train_df, val_df, test_df, stratify_col)
        
        return train_df, val_df, test_df
    
    def split_by_ratio(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide por proporción manteniendo orden temporal.
        
        Args:
            df: DataFrame completo
            train_ratio: Proporción para train
            val_ratio: Proporción para val
            test_ratio: Proporción para test
            
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Las proporciones deben sumar 1.0"
        
        # Ordenar por tiempo
        df = df.copy()
        df['_year_num'] = df[self.time_col].str.extract(r'(\d{4})').astype(int)
        df = df.sort_values('_year_num')
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Limpiar
        train_df = train_df.drop('_year_num', axis=1)
        val_df = val_df.drop('_year_num', axis=1)
        test_df = test_df.drop('_year_num', axis=1)
        
        print(f"[OK] Partición por ratio completada:")
        print(f"  Train: {len(train_df):,} filas ({train_ratio*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} filas ({val_ratio*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} filas ({test_ratio*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def _validate_stratification(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        stratify_col: str
    ):
        """Valida que las proporciones se mantengan."""
        original_dist = df[stratify_col].value_counts(normalize=True)
        train_dist = train_df[stratify_col].value_counts(normalize=True)
        val_dist = val_df[stratify_col].value_counts(normalize=True)
        test_dist = test_df[stratify_col].value_counts(normalize=True)
        
        print(f"\n[INFO] Distribución de '{stratify_col}':")
        print(f"  Original: {original_dist.to_dict()}")
        print(f"  Train: {train_dist.to_dict()}")
        print(f"  Val: {val_dist.to_dict()}")
        print(f"  Test: {test_dist.to_dict()}")
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "data"
    ):
        """
        Guarda las particiones en disco.
        
        Args:
            train_df, val_df, test_df: DataFrames a guardar
            output_dir: Directorio base
        """
        output_dir = Path(output_dir)
        
        # Crear directorios
        (output_dir / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "test").mkdir(parents=True, exist_ok=True)
        
        # Guardar CSVs
        train_df.to_csv(output_dir / "train" / "train.csv", index=False, encoding="utf-8")
        val_df.to_csv(output_dir / "val" / "val.csv", index=False, encoding="utf-8")
        test_df.to_csv(output_dir / "test" / "test.csv", index=False, encoding="utf-8")
        
        # Guardar metadata
        metadata = {
            **self.split_info,
            "time_col": self.time_col,
            "entity_col": self.entity_col,
        }
        
        with open(output_dir / "split_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Particiones guardadas en '{output_dir}'")


def split_titulados_data(
    df: pd.DataFrame,
    train_years: Optional[list] = None,
    val_years: Optional[list] = None,
    test_years: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Función de conveniencia para dividir datos de titulados.
    
    Por defecto usa:
    - Train: 2007-2022 (16 años)
    - Val: 2023 (1 año)
    - Test: 2024 (1 año)
    
    Args:
        df: DataFrame completo
        train_years, val_years, test_years: Años específicos (opcional)
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    if train_years is None:
        train_years = list(range(2007, 2023))  # 2007-2022
    if val_years is None:
        val_years = [2023]
    if test_years is None:
        test_years = [2024]
    
    splitter = TemporalSplitter(time_col="AÑO", entity_col="CÓDIGO INSTITUCIÓN")
    
    train_df, val_df, test_df = splitter.split_by_years(
        df,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        stratify_col="REGIÓN"
    )
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Ejemplo de uso
    from loader import load_titulados_data
    
    df, _ = load_titulados_data()
    train_df, val_df, test_df = split_titulados_data(df)
    
    # Guardar
    splitter = TemporalSplitter()
    splitter.save_splits(train_df, val_df, test_df)
