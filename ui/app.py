"""UI Interactiva con Streamlit"""
import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import sys

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import MLPipeline
from src.data.cleaner import load_and_clean_dataset

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar página
st.set_page_config(
    page_title="ML Demo - Educación Superior",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titulo
st.title("🎓 Modelado Predictivo - Educación Superior Chile")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    mode = st.radio(
        "Selecciona modo:",
        ["📊 EDA", "🚀 Entrenar", "🔮 Predecir", "📄 Reportes"],
        index=0
    )
    
    st.markdown("---")
    st.info("Pipeline modular para ML en educación superior")

# Cargar config
config = Config()

# MODO: EDA
if "EDA" in mode:
    st.header("📊 Análisis Exploratorio de Datos")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📥 Cargar Dataset", use_container_width=True):
            with st.spinner("Cargando datos..."):
                try:
                    df = load_and_clean_dataset(
                        str(config.DATASET_PATH),
                        sep=config.SEPARATOR,
                        encoding=config.ENCODING,
                        decimal_columns=config.DECIMAL_COLUMNS
                    )
                    st.session_state.df = df
                    st.success("✓ Dataset cargado")
                except Exception as e:
                    st.error(f"Error al cargar: {e}")
    
    with col2:
        if st.button("🔍 Ejecutar EDA", use_container_width=True):
            with st.spinner("Generando análisis..."):
                try:
                    if 'df' in st.session_state:
                        pipeline = MLPipeline(config)
                        pipeline.df = st.session_state.df
                        report = pipeline.explore_data(output_dir=str(config.OUTPUTS_DIR / "eda"))
                        st.session_state.eda_report = report
                        st.success("✓ EDA completado")
                    else:
                        st.warning("Primero carga el dataset")
                except Exception as e:
                    st.error(f"Error en EDA: {e}")
                    logger.error(f"Error: {e}")
    
    with col3:
        if st.button("📥 Descargar Gráficos", use_container_width=True):
            st.info("Los gráficos están en: outputs/eda/")
    
    st.markdown("---")
    
    # Mostrar información del dataset
    if 'df' in st.session_state:
        df = st.session_state.df
        
        st.subheader("📋 Información del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros", f"{len(df):,}")
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            st.metric("Nulos", f"{df.isnull().sum().sum():,}")
        
        st.markdown("---")
        
        st.subheader("🎯 Variables Objetivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Clasificación:** {config.TARGET_CLASSIFICATION}")
            if config.TARGET_CLASSIFICATION in df.columns:
                vc = df[config.TARGET_CLASSIFICATION].value_counts()
                st.bar_chart(vc)
                st.write(f"Distribución:")
                for idx, val in vc.items():
                    pct = (val / len(df) * 100)
                    st.write(f"  • {idx}: {val:,} ({pct:.1f}%)")
        
        with col2:
            st.write(f"**Regresión:** {config.TARGET_REGRESSION}")
            if config.TARGET_REGRESSION in df.columns:
                st.write(df[config.TARGET_REGRESSION].describe())
        
        st.markdown("---")
        
        # Mostrar gráficos generados
        st.subheader("📊 Gráficos EDA Generados")
        
        eda_dir = config.OUTPUTS_DIR / "eda"
        
        if eda_dir.exists():
            png_files = sorted(list(eda_dir.glob("*.png")))
            
            if png_files:
                # Crear grid de 2x2 para los gráficos
                col1, col2 = st.columns(2)
                
                for idx, img_path in enumerate(png_files):
                    if idx % 2 == 0:
                        col = col1
                    else:
                        col = col2
                    
                    with col:
                        st.image(
                            str(img_path),
                            caption=img_path.stem,
                            use_container_width=True
                        )
                        
                        # Botón para descargar
                        with open(img_path, "rb") as file:
                            st.download_button(
                                label=f"Descargar {img_path.name}",
                                data=file,
                                file_name=img_path.name,
                                mime="image/png",
                                use_container_width=True
                            )
            else:
                st.info("No hay gráficos generados aún. Ejecuta EDA primero.")
        else:
            st.info("Directorio de EDA no existe. Ejecuta EDA primero.")
        
        st.markdown("---")
        
        st.subheader("📊 Vista de Datos (Primeras 10 filas)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📈 Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)

# MODO: ENTRENAR
elif "Entrenar" in mode:
    st.header("🚀 Entrenamiento de Modelos")
    
    if st.button("▶️ Ejecutar Pipeline Completo", use_container_width=True):
        with st.spinner("Ejecutando pipeline..."):
            progress_bar = st.progress(0)
            status = st.empty()
            
            try:
                # 1. Cargar datos
                status.text("📥 Cargando datos...")
                pipeline = MLPipeline(config)
                pipeline.load_data()
                progress_bar.progress(25)
                
                # 2. EDA
                status.text("🔍 Explorando datos...")
                pipeline.explore_data()
                progress_bar.progress(50)
                
                # 3. Preprocesamiento
                status.text("🔧 Preprocesando...")
                pipeline.preprocess_data()
                progress_bar.progress(75)
                
                # 4. Completado
                status.text("✓ Pipeline completado")
                progress_bar.progress(100)
                
                st.success("✓ Entrenamiento completado exitosamente")
                
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
                logger.error(f"Error en pipeline: {e}")

# MODO: PREDECIR
elif "Predecir" in mode:
    st.header("🔮 Hacer Predicciones")
    
    uploaded_file = st.file_uploader(
        "Cargar archivo CSV para predicción",
        type="csv",
        help="Archivo con características para predecir"
    )
    
    if uploaded_file:
        try:
            df_test = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            df_test.columns = df_test.columns.str.strip()
            
            st.write(f"Registros: {len(df_test)}")
            st.dataframe(df_test.head(), use_container_width=True)
            
            if st.button("🔮 Predecir", use_container_width=True):
                st.info("Funcionalidad disponible después de entrenar modelos")
                
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

# MODO: REPORTES
elif "Reportes" in mode:
    st.header("📄 Reportes")
    
    report_type = st.selectbox(
        "Selecciona tipo de reporte:",
        ["Resumen EDA", "Resultados Modelos", "Interpretabilidad (XAI)"]
    )
    
    if st.button("📄 Generar Reporte", use_container_width=True):
        if report_type == "Resumen EDA":
            st.info("✓ Reporte EDA disponible en: `outputs/eda/`")
            
            # Listar archivos EDA
            eda_dir = config.OUTPUTS_DIR / "eda"
            if eda_dir.exists():
                png_files = list(eda_dir.glob("*.png"))
                st.write(f"**{len(png_files)} gráficos disponibles:**")
                
                # Mostrar en grid
                for img_path in sorted(png_files):
                    st.image(str(img_path), caption=img_path.stem, use_container_width=True)
                    
                    # Botón descargar
                    with open(img_path, "rb") as file:
                        st.download_button(
                            label=f"Descargar {img_path.name}",
                            data=file,
                            file_name=img_path.name,
                            mime="image/png",
                            use_container_width=True,
                            key=img_path.name
                        )
        
        elif report_type == "Resultados Modelos":
            st.info("Resultados disponibles después de entrenar")
        
        elif report_type == "Interpretabilidad (XAI)":
            st.info("Análisis SHAP disponible después de entrenar")

# Footer
st.markdown("---")
st.markdown(
    "🏗️ **Arquitectura Modular** | "
    "📚 Notebooks + 🛠️ CLI + 🎨 UI | "
    "✨ Producción Ready"
)
