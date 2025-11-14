"""UI Fases 1 y 2 (Objetivos + EDA)"""
import streamlit as st
import pandas as pd
from pathlib import Path
import logging
import sys

# Añadir raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from src.eda import cargar_csv, eda_minimo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fases 1-2 ML", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 Proyecto ML Educación Superior - Fases 1 y 2")
st.markdown("---")

config = Config()

# Estado de sesión
if 'df' not in st.session_state:
    st.session_state.df = None
if 'eda_result' not in st.session_state:
    st.session_state.eda_result = None

fase = st.sidebar.radio("Selecciona Fase:", ["Fase 1 - Objetivos", "Fase 2 - EDA"], index=0)

st.sidebar.markdown("---")
st.sidebar.write(f"Dataset: {config.DATASET_PATH}")
st.sidebar.write(f"Targets: {config.TARGET_CLASSIFICATION} / {config.TARGET_REGRESSION}")
if st.sidebar.button("🧹 Limpiar artefactos (clean.sh)"):
    import subprocess
    try:
        subprocess.run(["bash", "clean.sh"], check=True)
        st.sidebar.success("Limpieza completada")
    except Exception as e:
        st.sidebar.error(f"Error al limpiar: {e}")



# ============ FASE 1 ============
if fase.startswith("Fase 1"):
    st.header("Fase 1: Comprensión del Caso y Objetivos")
    ficha_path = Path("docs/objetivo_ficha.md")
    if ficha_path.exists():
        st.subheader("Ficha de Objetivos")
        st.markdown(ficha_path.read_text(encoding='utf-8'))
    else:
        st.warning("No existe docs/objetivo_ficha.md")

    if st.button("🔎 Validar Config", use_container_width=True):
        st.success("Configuración cargada")
        st.json({
            "TARGET_CLASSIFICATION": config.TARGET_CLASSIFICATION,
            "TARGET_REGRESSION": config.TARGET_REGRESSION,
            "METRICS": config.METRICS,
            "RISKS": config.RISKS
        })
    st.info("Criterios Fase 1: ficha presente + targets + métricas + riesgos mostrados.")

# ============ FASE 2 ============
elif fase.startswith("Fase 2"):
    st.header("Fase 2: EDA Mínimo")
    col_load, col_run, col_view = st.columns(3)

    with col_load:
        if st.button("📥 Cargar Dataset", use_container_width=True):
            try:
                df = cargar_csv(config.DATASET_PATH)
                st.success(f"Dataset cargado: {df.shape}")
                # Crear MODALIDAD_BIN si falta y existe MODALIDAD
                if config.TARGET_CLASSIFICATION not in df.columns and 'MODALIDAD' in df.columns:
                    tmp = df['MODALIDAD'].astype(str).str.strip().str.lower()
                    df['MODALIDAD_BIN'] = tmp.apply(lambda v: 1 if v.startswith('presencial') else 0)
                    st.info("Columna MODALIDAD_BIN creada (Presencial=1, otros=0) para generar decision_metricas.txt")
                st.session_state.df = df
            except Exception as e:
                st.error(f"Error cargando dataset: {e}")

    with col_run:
        if st.button("🔍 Ejecutar EDA", use_container_width=True):
            if st.session_state.df is None:
                st.warning("Primero carga el dataset")
            else:
                Path("outputs/eda/resumen").mkdir(parents=True, exist_ok=True)
                Path("outputs/eda/figures").mkdir(parents=True, exist_ok=True)
                st.session_state.eda_result = eda_minimo(st.session_state.df, objetivo=config.TARGET_CLASSIFICATION, no_show=True)
                # También generar gráficos Fase 1 para compatibilidad
                import subprocess
                try:
                    subprocess.run(["./venv/bin/python", "scripts/execute_pipeline.py", "--phase", "1"], check=True)
                    st.success("EDA completado + Gráficos Fase 1 generados")
                except Exception as e:
                    st.success("EDA completado")
                    st.warning(f"No se pudieron generar los gráficos Fase 1: {e}")

    with col_view:
        if st.button("📄 Ver Artefactos", use_container_width=True):
            if st.session_state.eda_result:
                st.write(st.session_state.eda_result)
            else:
                st.info("Ejecuta el EDA primero")

    st.markdown("---")


    df = st.session_state.df
    if df is not None:
        st.subheader("Resumen rápido")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Filas", df.shape[0])
        colB.metric("Columnas", df.shape[1])
        colC.metric("Nulos", int(df.isna().sum().sum()))
        if config.TARGET_CLASSIFICATION in df.columns and df[config.TARGET_CLASSIFICATION].nunique()>1:
            ir = df[config.TARGET_CLASSIFICATION].value_counts().max()/df[config.TARGET_CLASSIFICATION].value_counts().min()
            colD.metric("Imbalance Ratio", f"{ir:.2f}")
        else:
            colD.metric("Imbalance Ratio", "-")

        st.subheader("Variable Objetivo")
        if config.TARGET_CLASSIFICATION in df.columns:
            vc = df[config.TARGET_CLASSIFICATION].value_counts()
            st.bar_chart(vc)
            for k,v in vc.items():
                st.write(f"• Clase {k}: {v} ({v/len(df)*100:.1f}%)")

        st.subheader("Primeras Filas")
        st.dataframe(df.head(), use_container_width=True)

    st.subheader("Artefactos Generados")



    artefacts = [
            "resumen_columnas.csv",
            "resumen_columnas_ordenado.csv",
            "top10_faltantes.csv",
            "descriptivos_numericos.csv",
            "decision_metricas.txt"
        ]
    for a in artefacts:
        p = Path(f"outputs/eda/resumen/{a}")
        if p.exists():
            st.write(f"✅ {a}")
            if a.endswith('.csv'):
                st.dataframe(pd.read_csv(p), use_container_width=True)
            elif a.endswith('.txt'):
                st.code(p.read_text(encoding='utf-8'), language='text')
        else:
            st.write(f"⌛ {a} pendiente")


    # Gráficos generados en data/processed (Fase 1)
    st.subheader("Gráficos en data/processed")
    proc_dir = Path("data/processed")
    if proc_dir.exists():
        proc_imgs = [
            proc_dir / '01_values_count.png',
            proc_dir / '02_edad_distribucion.png',
            proc_dir / '03_distribution_program.png',
            proc_dir / '04_correlation_matrix.png',
            proc_dir / '05_missing_values.png',
            proc_dir / '06_outliers_detection.png',
        ]
        shown_any=False
        for img in proc_imgs:
            if img.exists():
                st.image(str(img), caption=img.name, use_container_width=True)
                shown_any=True
        if not shown_any:
            st.info("No hay imágenes en data/processed aún")
    else:
        st.info("Directorio data/processed no existe")
    

    st.subheader("Otros Gráficos")
    fig_dir = Path("outputs/eda/figures")
    if fig_dir.exists():
        imgs = list(fig_dir.glob("*.png"))
        if imgs:
            for img in imgs:
                st.image(str(img), caption=img.stem, use_container_width=True)
        else:
            st.info("No hay gráficos todavía")
    else:
        st.info("Directorio de figuras no existe")

    st.info("Criterios Fase 2: CSV + gráficos + decisión métricas presentes.")

st.markdown("---")
