"""
Módulo de preprocesamiento de datos.

Fase 1: Datos y particiones
- Imputación de valores faltantes
- Estandarización (z-score)
- Normalización (min-max)
- Tratamiento de outliers
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class DataPreprocessor:
    """
    Preprocesador de datos evitando data leakage.
    
    IMPORTANTE: Ajustar solo en train, aplicar en val/test.
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoding_maps = {}
        self.preprocessing_info = {}
        
    def impute_missing(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Imputa valores faltantes.
        
        Args:
            df: DataFrame a procesar
            strategy: Dict {columna: estrategia}
                Estrategias: 'mean', 'median', 'mode', 'constant', 'forward', 'backward'
            fit: Si True, calcula estadísticos. Si False, usa ya calculados.
            
        Returns:
            DataFrame con valores imputados
        """
        df = df.copy()
        
        for col, method in strategy.items():
            if col not in df.columns:
                warnings.warn(f"Columna '{col}' no existe")
                continue
                
            if fit:
                if method == 'mean':
                    self.imputers[col] = df[col].mean()
                elif method == 'median':
                    self.imputers[col] = df[col].median()
                elif method == 'mode':
                    self.imputers[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                elif method == 'constant':
                    self.imputers[col] = 0  # O el valor que definas
                    
            # Aplicar imputación
            if method in ['mean', 'median', 'mode', 'constant']:
                df[col].fillna(self.imputers[col], inplace=True)
            elif method == 'forward':
                df[col].fillna(method='ffill', inplace=True)
            elif method == 'backward':
                df[col].fillna(method='bfill', inplace=True)
                
        return df
    
    def standardize(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Estandariza columnas (z-score).
        
        Fórmula: z = (x - media) / desviación
        
        Args:
            df: DataFrame a procesar
            columns: Columnas numéricas a estandarizar
            fit: Si True, ajusta en estos datos. Si False, usa previo.
            
        Returns:
            DataFrame con columnas estandarizadas
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Columna '{col}' no existe")
                continue
                
            if fit:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[f"{col}_standard"] = scaler
            else:
                if f"{col}_standard" not in self.scalers:
                    warnings.warn(f"No hay scaler ajustado para '{col}'")
                    continue
                scaler = self.scalers[f"{col}_standard"]
                df[col] = scaler.transform(df[[col]])
                
        return df
    
    def normalize(
        self,
        df: pd.DataFrame,
        columns: List[str],
        feature_range: Tuple[float, float] = (0, 1),
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normaliza columnas (min-max).
        
        Fórmula: (x - min) / (max - min)
        
        Args:
            df: DataFrame a procesar
            columns: Columnas numéricas a normalizar
            feature_range: Rango objetivo (min, max)
            fit: Si True, ajusta en estos datos. Si False, usa previo.
            
        Returns:
            DataFrame con columnas normalizadas
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Columna '{col}' no existe")
                continue
                
            if fit:
                scaler = MinMaxScaler(feature_range=feature_range)
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[f"{col}_minmax"] = scaler
            else:
                if f"{col}_minmax" not in self.scalers:
                    warnings.warn(f"No hay scaler ajustado para '{col}'")
                    continue
                scaler = self.scalers[f"{col}_minmax"]
                df[col] = scaler.transform(df[[col]])
                
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'winsorize',
        percentiles: Tuple[float, float] = (0.01, 0.99),
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Trata valores atípicos (outliers).
        
        Args:
            df: DataFrame a procesar
            columns: Columnas a tratar
            method: 'winsorize' (recortar), 'clip' (limitar), 'remove' (eliminar)
            percentiles: Percentiles inferior y superior
            fit: Si True, calcula límites. Si False, usa previo.
            
        Returns:
            DataFrame con outliers tratados
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Columna '{col}' no existe")
                continue
                
            if fit:
                lower = df[col].quantile(percentiles[0])
                upper = df[col].quantile(percentiles[1])
                self.preprocessing_info[f"{col}_outlier_bounds"] = (lower, upper)
            else:
                if f"{col}_outlier_bounds" not in self.preprocessing_info:
                    warnings.warn(f"No hay límites ajustados para '{col}'")
                    continue
                lower, upper = self.preprocessing_info[f"{col}_outlier_bounds"]
                
            if method == 'winsorize' or method == 'clip':
                df[col] = df[col].clip(lower, upper)
            elif method == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                
        return df
    
    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige tipos de datos del dataset de titulados.
        
        - Convierte promedios de edad de texto a float
        - Extrae año numérico
        - Convierte columnas numéricas mal clasificadas
        
        Args:
            df: DataFrame a corregir
            
        Returns:
            DataFrame con tipos corregidos
        """
        df = df.copy()
        
        # Extraer año numérico
        if 'AÑO' in df.columns:
            df['año_numerico'] = df['AÑO'].str.extract(r'(\d{4})').astype(int)
        
        # Corregir promedios de edad (comas por puntos)
        edad_cols = [col for col in df.columns if 'PROMEDIO EDAD' in col]
        for col in edad_cols:
            if df[col].dtype == 'object':
                # Reemplazar comas por puntos y convertir
                df[col] = df[col].str.replace(',', '.').astype(float, errors='ignore')
        
        # Asegurar que TOTAL TITULACIONES sea numérico
        if 'TOTAL TITULACIONES' in df.columns:
            df['TOTAL TITULACIONES'] = pd.to_numeric(
                df['TOTAL TITULACIONES'], errors='coerce'
            )
        
        print("[OK] Tipos de datos corregidos")
        return df
    
    def drop_high_missing_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Elimina columnas con muchos valores faltantes.
        
        Args:
            df: DataFrame a limpiar
            threshold: Umbral de % faltantes (0-1)
            
        Returns:
            DataFrame sin columnas con muchos faltantes
        """
        df = df.copy()
        
        missing_pct = df.isna().mean()
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            print(f"[INFO] Eliminando {len(cols_to_drop)} columnas con >{threshold*100}% faltantes:")
            for col in cols_to_drop:
                print(f"  - {col}: {missing_pct[col]*100:.1f}% faltantes")
            df = df.drop(columns=cols_to_drop)
        else:
            print(f"[INFO] No hay columnas con >{threshold*100}% faltantes")
            
        return df


def preprocess_titulados_data(
    df: pd.DataFrame,
    fit: bool = True,
    preprocessor: Optional[DataPreprocessor] = None
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Pipeline completo de preprocesamiento para datos de titulados.
    
    Args:
        df: DataFrame a procesar
        fit: Si True, ajusta parámetros. Si False, usa ya ajustados.
        preprocessor: Preprocesador ya ajustado (para val/test)
        
    Returns:
        Tuple (df_procesado, preprocessor)
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor()
    
    # 1. Corregir tipos
    df = preprocessor.fix_data_types(df)
    
    # 2. Eliminar columnas con muchos faltantes (solo en fit)
    if fit:
        df = preprocessor.drop_high_missing_columns(df, threshold=0.95)
    
    # 3. Imputar valores faltantes
    imputation_strategy = {
        'DURACIÓN ESTUDIO CARRERA': 'median',
        'DURACIÓN TOTAL DE LA CARRERA': 'median',
        'TITULACIONES MUJERES POR PROGRAMA': 'constant',
        'TITULACIONES HOMBRES POR PROGRAMA': 'constant',
    }
    df = preprocessor.impute_missing(df, imputation_strategy, fit=fit)
    
    # 4. Tratar outliers en titulaciones (usar log en su lugar)
    if 'TOTAL TITULACIONES' in df.columns:
        df['log_titulaciones'] = np.log1p(df['TOTAL TITULACIONES'])
    
    print(f"[OK] Preprocesamiento {'ajustado y aplicado' if fit else 'aplicado'}")
    return df, preprocessor


if __name__ == "__main__":
    # Ejemplo de uso
    from loader import load_titulados_data
    from splitter import split_titulados_data
    
    df, _ = load_titulados_data()
    train_df, val_df, test_df = split_titulados_data(df)
    
    # Preprocesar train (fit=True)
    train_processed, preprocessor = preprocess_titulados_data(train_df, fit=True)
    
    # Preprocesar val y test (fit=False, usa parámetros de train)
    val_processed, _ = preprocess_titulados_data(val_df, fit=False, preprocessor=preprocessor)
    test_processed, _ = preprocess_titulados_data(test_df, fit=False, preprocessor=preprocessor)
    
    print("\n[OK] Train, Val y Test preprocesados sin data leakage")
