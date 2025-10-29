"""
Módulo de ingeniería de características (features).

Fase 2: Features engineering
- Features temporales (rezagos, rolling, variación porcentual)
- Features categóricas (encoding, agregaciones)
- Features derivadas específicas del dominio
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings


class FeatureEngineer:
    """Creador de features para datos de titulados."""
    
    def __init__(self):
        self.feature_names = []
        self.feature_metadata = {}
        
    def create_temporal_features(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_cols: List[str],
        lags: List[int] = [1, 2, 3],
        rolling_windows: List[int] = [3, 5],
        pct_change_periods: List[int] = [1, 3]
    ) -> pd.DataFrame:
        """
        Crea features temporales.
        
        Args:
            df: DataFrame con columna año_numerico
            value_cols: Columnas de valores a procesar
            group_cols: Columnas para agrupar (ej: institución, región)
            lags: Lista de rezagos a crear
            rolling_windows: Lista de ventanas para promedios móviles
            pct_change_periods: Períodos para variación porcentual
            
        Returns:
            DataFrame con nuevas features temporales
        """
        df = df.copy()
        
        if 'año_numerico' not in df.columns:
            raise ValueError("Se requiere columna 'año_numerico'")
        
        # Ordenar por tiempo
        df = df.sort_values(['año_numerico'] + group_cols)
        
        for col in value_cols:
            if col not in df.columns:
                continue
                
            # Rezagos (lags)
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df.groupby(group_cols)[col].shift(lag)
                self.feature_names.append(lag_col)
                self.feature_metadata[lag_col] = {
                    "type": "temporal_lag",
                    "base_col": col,
                    "lag": lag
                }
            
            # Promedios móviles (rolling)
            for window in rolling_windows:
                roll_col = f"{col}_rolling{window}"
                df[roll_col] = df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                self.feature_names.append(roll_col)
                self.feature_metadata[roll_col] = {
                    "type": "rolling_mean",
                    "base_col": col,
                    "window": window
                }
            
            # Variación porcentual
            for period in pct_change_periods:
                pct_col = f"{col}_pctchange{period}"
                df[pct_col] = df.groupby(group_cols)[col].pct_change(periods=period) * 100
                self.feature_names.append(pct_col)
                self.feature_metadata[pct_col] = {
                    "type": "pct_change",
                    "base_col": col,
                    "period": period
                }
        
        print(f"[OK] Creadas {len(self.feature_names)} features temporales")
        return df
    
    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_col: str,
        agg_funcs: List[str] = ['sum', 'mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Crea features de agregación.
        
        Args:
            df: DataFrame
            group_cols: Columnas para agrupar
            agg_col: Columna a agregar
            agg_funcs: Funciones de agregación
            
        Returns:
            DataFrame con features de agregación
        """
        df = df.copy()
        
        for func in agg_funcs:
            agg_name = f"{agg_col}_{func}_by_{'_'.join(group_cols)}"
            
            if func == 'sum':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('sum')
            elif func == 'mean':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('mean')
            elif func == 'std':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('std')
            elif func == 'min':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('min')
            elif func == 'max':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('max')
            elif func == 'count':
                df[agg_name] = df.groupby(group_cols)[agg_col].transform('count')
                
            self.feature_names.append(agg_name)
            self.feature_metadata[agg_name] = {
                "type": "aggregation",
                "base_col": agg_col,
                "group_by": group_cols,
                "function": func
            }
        
        print(f"[OK] Creadas {len(agg_funcs)} features de agregación")
        return df
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_col: str,
        denominator_col: str,
        name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crea features de ratio.
        
        Args:
            df: DataFrame
            numerator_col: Columna numerador
            denominator_col: Columna denominador
            name: Nombre de la nueva feature (opcional)
            
        Returns:
            DataFrame con feature de ratio
        """
        df = df.copy()
        
        if name is None:
            name = f"{numerator_col}_div_{denominator_col}"
        
        # Evitar división por cero
        df[name] = df[numerator_col] / (df[denominator_col].replace(0, np.nan))
        
        self.feature_names.append(name)
        self.feature_metadata[name] = {
            "type": "ratio",
            "numerator": numerator_col,
            "denominator": denominator_col
        }
        
        return df
    
    def create_categorical_features(
        self,
        df: pd.DataFrame,
        stem_areas: List[str] = None,
        salud_areas: List[str] = None
    ) -> pd.DataFrame:
        """
        Crea features categóricas específicas del dominio.
        
        Args:
            df: DataFrame
            stem_areas: Lista de áreas STEM
            salud_areas: Lista de áreas de salud
            
        Returns:
            DataFrame con features categóricas
        """
        df = df.copy()
        
        # STEM
        if stem_areas is None:
            stem_areas = [
                'Tecnología', 'Ciencias Básicas', 'Ingeniería'
            ]
        if 'ÁREA DEL CONOCIMIENTO' in df.columns:
            df['es_STEM'] = df['ÁREA DEL CONOCIMIENTO'].isin(stem_areas).astype(int)
            self.feature_names.append('es_STEM')
        
        # Salud
        if salud_areas is None:
            salud_areas = ['Salud']
        if 'ÁREA DEL CONOCIMIENTO' in df.columns:
            df['es_salud'] = df['ÁREA DEL CONOCIMIENTO'].isin(salud_areas).astype(int)
            self.feature_names.append('es_salud')
        
        # Tipo institución
        if 'CLASIFICACIÓN INSTITUCIÓN NIVEL 1' in df.columns:
            df['es_universidad'] = (
                df['CLASIFICACIÓN INSTITUCIÓN NIVEL 1'] == 'Universidades'
            ).astype(int)
            self.feature_names.append('es_universidad')
        
        # Nivel
        if 'NIVEL GLOBAL' in df.columns:
            df['es_postgrado'] = (
                df['NIVEL GLOBAL'].isin(['Posgrado', 'Postítulo'])
            ).astype(int)
            self.feature_names.append('es_postgrado')
        
        # Modalidad
        if 'MODALIDAD' in df.columns:
            df['es_presencial'] = (
                df['MODALIDAD'] == 'Presencial'
            ).astype(int)
            self.feature_names.append('es_presencial')
        
        # Pandemia
        if 'año_numerico' in df.columns:
            df['es_pandemia'] = (
                df['año_numerico'].isin([2020, 2021])
            ).astype(int)
            self.feature_names.append('es_pandemia')
        
        print(f"[OK] Creadas {6} features categóricas")
        return df
    
    def create_gender_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de género.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame con features de género
        """
        df = df.copy()
        
        if 'TITULACIONES MUJERES POR PROGRAMA' in df.columns and \
           'TITULACIONES HOMBRES POR PROGRAMA' in df.columns and \
           'TOTAL TITULACIONES' in df.columns:
            
            # Ratios
            df['ratio_mujeres'] = (
                df['TITULACIONES MUJERES POR PROGRAMA'] / df['TOTAL TITULACIONES']
            ).fillna(0.5)
            
            df['ratio_hombres'] = (
                df['TITULACIONES HOMBRES POR PROGRAMA'] / df['TOTAL TITULACIONES']
            ).fillna(0.5)
            
            # Paridad (0 = paridad perfecta)
            df['paridad_genero'] = np.abs(0.5 - df['ratio_mujeres'])
            
            # Dominancia
            df['dominio_mujeres'] = (df['ratio_mujeres'] > 0.6).astype(int)
            df['dominio_hombres'] = (df['ratio_hombres'] > 0.6).astype(int)
            df['dominio_neutro'] = (
                (df['ratio_mujeres'] >= 0.4) & (df['ratio_mujeres'] <= 0.6)
            ).astype(int)
            
            features = ['ratio_mujeres', 'ratio_hombres', 'paridad_genero',
                       'dominio_mujeres', 'dominio_hombres', 'dominio_neutro']
            self.feature_names.extend(features)
            
            print(f"[OK] Creadas {len(features)} features de género")
        
        return df
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de features creadas.
        
        Returns:
            DataFrame con resumen
        """
        if not self.feature_metadata:
            return pd.DataFrame()
        
        summary = pd.DataFrame.from_dict(self.feature_metadata, orient='index')
        summary.index.name = 'feature_name'
        summary = summary.reset_index()
        
        return summary


def create_titulados_features(
    df: pd.DataFrame,
    include_temporal: bool = True,
    include_aggregations: bool = True,
    include_ratios: bool = True,
    include_categorical: bool = True,
    include_gender: bool = True
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Pipeline completo de feature engineering para titulados.
    
    Args:
        df: DataFrame preprocesado
        include_*: Flags para incluir cada tipo de features
        
    Returns:
        Tuple (df_con_features, feature_engineer)
    """
    engineer = FeatureEngineer()
    
    # Features categóricas (primero, las necesitamos para agrupar)
    if include_categorical:
        df = engineer.create_categorical_features(df)
    
    # Features de género
    if include_gender:
        df = engineer.create_gender_features(df)
    
    # Features temporales
    if include_temporal and 'año_numerico' in df.columns:
        df = engineer.create_temporal_features(
            df,
            value_cols=['TOTAL TITULACIONES', 'ratio_mujeres'],
            group_cols=['CÓDIGO INSTITUCIÓN'],
            lags=[1, 2],
            rolling_windows=[3],
            pct_change_periods=[1]
        )
    
    # Features de agregación
    if include_aggregations:
        # Por región y año
        df = engineer.create_aggregation_features(
            df,
            group_cols=['REGIÓN', 'año_numerico'],
            agg_col='TOTAL TITULACIONES',
            agg_funcs=['sum', 'mean']
        )
    
    # Features de ratio
    if include_ratios and 'DURACIÓN ESTUDIO CARRERA' in df.columns:
        df = engineer.create_ratio_features(
            df,
            'DURACIÓN TOTAL DE LA CARRERA',
            'DURACIÓN ESTUDIO CARRERA',
            'ratio_duracion'
        )
    
    print(f"\n[OK] Feature engineering completado")
    print(f"Total features creadas: {len(engineer.feature_names)}")
    
    return df, engineer


if __name__ == "__main__":
    # Ejemplo de uso
    from data.loader import load_titulados_data
    from data.preprocessor import preprocess_titulados_data
    
    df, _ = load_titulados_data()
    df_processed, _ = preprocess_titulados_data(df, fit=True)
    df_features, engineer = create_titulados_features(df_processed)
    
    print("\nFeatures creadas:")
    print(engineer.get_feature_summary().head(20))
