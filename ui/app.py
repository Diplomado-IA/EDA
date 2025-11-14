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

st.set_page_config(page_title="ML", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 Proyecto ML Educación Superior")
st.markdown("---")

config = Config()

# Estado de sesión
if 'df' not in st.session_state:
    st.session_state.df = None
if 'eda_result' not in st.session_state:
    st.session_state.eda_result = None

fase = st.sidebar.radio("Selecciona Fase:", ["Fase 1 - Objetivos", "Fase 2 - EDA", "Fase 3 - Preprocesamiento", "Informes"], index=0)

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
    st.caption("Definición de objetivos del modelo: qué predecir, con qué métricas y riesgos; alinea negocio con aprendizaje automático.")
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
    st.caption("EDA: Análisis Exploratorio de Datos; inspección cualitativa y cuantitativa para entender distribución, valores faltantes y relaciones.")
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
                # Mostrar artefactos generados inmediatamente
                artefacts = [
                    "resumen_columnas.csv",
                    "resumen_columnas_ordenado.csv",
                    "top10_faltantes.csv",
                    "descriptivos_numericos.csv",
                    "decision_metricas.txt",
                ]
                for a in artefacts:
                    p = Path(f"outputs/eda/resumen/{a}")
                    if p.exists():
                        st.caption(f"Guardado en: {p.resolve()}")
                        if a.endswith('.csv'):
                            st.dataframe(pd.read_csv(p).head(20), use_container_width=True)
                        else:
                            st.code(p.read_text(encoding='utf-8')[:2000], language='text')

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
            st.caption(f"Definición: {config.TARGET_CLASSIFICATION} — Clase 1 = Presencial; Clase 0 = No presencial/otras modalidades.")
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

# ============ FASE 3 ============
elif fase.startswith("Fase 3"):
    st.header("Fase 3: Preprocesamiento Interactivo")
    st.markdown("""
    En esta fase preparas el dataset para modelado de forma guiada, evitando fugas de información.
    Las acciones modifican el estado en memoria (st.session_state) y algunas generan artefactos en data/processed u outputs/eda/resumen.
    Sugerencia: ejecuta las secciones en orden (1→6). Si rehaces un paso, revisa impactos en los siguientes.
    """)
    from src.preprocessing.clean import (
        _ensure_modalidad_bin,
        _coerce_regression_target,
        impute_values,
        temporal_split,
        scale_numeric,
        one_hot_encode,
        engineer_features,
        correlation_matrix,
        compute_vif,
        select_features,
        save_datasets,
    )
    if st.session_state.df is None:
        st.warning("Primero ejecuta la Fase 2 para cargar el dataset")
    else:
        df = st.session_state.df.copy()
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1) Limpieza",
            "2) Split Temporal",
            "3) Codificar",
            "4) Features",
            "5) Correlación & VIF",
            "6) Selección & Guardado",
        ])
        with tab1:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: asegurar objetivos bien definidos y datos limpios/consistentes antes de dividir y transformar.
            - Comportamiento esperado: reducción de nulos, creación/normalización de objetivos; no escribe archivos todavía.
            - Botones:
              - Crear MODALIDAD_BIN y convertir objetivo: crea la variable binaria a partir de MODALIDAD (Presencial=1, otras=0) y fuerza el target de regresión a numérico.
              - Imputar nulos según Config: aplica estrategias de imputación (numéricas/categóricas) definidas en Config y actualiza el DataFrame en sesión.
            Notas: operación idempotente; puedes ejecutar más de una vez sin duplicar columnas.
            """)
            colA, colB = st.columns(2)
            if colA.button("Crear MODALIDAD_BIN y convertir objetivo", use_container_width=True):
                df = _ensure_modalidad_bin(df)
                df = _coerce_regression_target(df)
                st.session_state.df = df
                st.success("MODALIDAD_BIN y objetivo convertidos")
                st.write(df[[c for c in ["MODALIDAD", "MODALIDAD_BIN", "PROMEDIO EDAD PROGRAMA "] if c in df.columns]].head())
            if colB.button("Imputar nulos según Config", use_container_width=True):
                df = impute_values(df)
                st.session_state.df = df
                st.success("Imputación completada")
                # Mostrar resultados en distribución 50-50
                left, right = st.columns([1,1])
                with left:
                    st.caption("Imputación: numéricas=mediana, categóricas=moda (según Config).")
                with right:
                    nulls = df.isnull().sum().sort_values(ascending=False).head(10).to_frame("nulos")
                    st.dataframe(nulls, use_container_width=True)
        with tab2:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: crear un corte temporal para evaluar generalización y evitar data leakage.
            - Comportamiento esperado: se crean train_df y test_df en sesión, respetando años de Config; no se escriben archivos.
            - Botón:
              - Split temporal (train/test): separa por año (train<=2018, gap 2019, test>=2020) y muestra rangos + proporción de la clase objetivo.
            Notas: requiere que exista la columna AÑO.
            """)
            if st.button("Split temporal (train/test)", use_container_width=True):
                try:
                    # Snapshot antes
                    before_cols = st.session_state.df.columns.tolist()
                    before_nrows = len(st.session_state.df)
                    before_nulls = int(st.session_state.df.isnull().sum().sum())
                    before_year_dtype = st.session_state.df['AÑO'].dtype if 'AÑO' in st.session_state.df.columns else 'NA'

                    train_df, test_df = temporal_split(st.session_state.df)
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df

                    # Snapshot después
                    after_cols_train = train_df.columns.tolist()
                    after_cols_test = test_df.columns.tolist()

                    st.success(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
                    st.write("Rango:", f"Train {train_df['AÑO'].min()}-{train_df['AÑO'].max()} | Test {test_df['AÑO'].min()}-{test_df['AÑO'].max()}")
                    st.caption("Los gráficos comparan la distribución de la variable objetivo MODALIDAD_BIN entre train y test. Clase 1 = Presencial; Clase 0 = No presencial/otras modalidades. Sirve para detectar cambios de proporción tras el split temporal.")

                    with st.expander("Detalles antes/después"):
                        st.write("Antes del split:")
                        st.write({"filas": before_nrows, "nulos_totales": before_nulls, "cols": len(before_cols), "dtype_AÑO": str(before_year_dtype)})
                        st.write("Después del split:")
                        st.write({"train_filas": len(train_df), "test_filas": len(test_df), "train_cols": len(after_cols_train), "test_cols": len(after_cols_test)})
                        if config.TARGET_CLASSIFICATION in train_df.columns:
                            vc_train = train_df[config.TARGET_CLASSIFICATION].value_counts()
                            vc_test = test_df[config.TARGET_CLASSIFICATION].value_counts()
                            st.write("Distribución objetivo (Train):", vc_train.to_dict())
                            st.write("Distribución objetivo (Test):", vc_test.to_dict())

                    if config.TARGET_CLASSIFICATION in train_df.columns:
                        col1, col2 = st.columns(2)
                        col1.bar_chart(train_df[config.TARGET_CLASSIFICATION].value_counts())
                        col2.bar_chart(test_df[config.TARGET_CLASSIFICATION].value_counts())
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc(), language='text')
        with tab3:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: preparar variables para modelos (normalizar numéricas y codificar categóricas) sin mezclar información de test.
            - Comportamiento esperado: el número de columnas puede cambiar; se guardan detalles en outputs/eda/resumen.
            - Botones:
              - Estandarizar numéricas: ajusta StandardScaler con train y aplica a test (sin fuga); guarda scaler_info.txt.
              - One-Hot Encoding: crea dummies consistentes en train y test; guarda one_hot_columns.txt.
            Notas: excluye las columnas objetivo de cualquier transformación.
            """)
            if st.session_state.get("train_df") is None:
                st.info("Ejecuta el split primero")
            else:
                colA, colB = st.columns(2)
                if colA.button("Estandarizar numéricas", use_container_width=True):
                    train_df, test_df, info = scale_numeric(st.session_state.train_df, st.session_state.test_df, exclude=[config.TARGET_CLASSIFICATION, config.TARGET_REGRESSION])
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df
                    Path("outputs/eda/resumen").mkdir(parents=True, exist_ok=True)
                    scaler_path = Path("outputs/eda/resumen/scaler_info.txt")
                    scaler_path.write_text(str(info), encoding="utf-8")
                    st.success("Escalado aplicado (StandardScaler)")
                    st.caption(f"Guardado en: {scaler_path.resolve()}")
                    st.code(scaler_path.read_text(encoding='utf-8')[:2000], language='text')
                if colB.button("One-Hot Encoding", use_container_width=True):
                    train_enc, test_enc, cols = one_hot_encode(st.session_state.train_df, st.session_state.test_df)
                    st.session_state.train_df = train_enc
                    st.session_state.test_df = test_enc
                    oh_path = Path("outputs/eda/resumen/one_hot_columns.txt")
                    oh_path.write_text("\n".join(cols), encoding="utf-8")
                    st.success(f"One-Hot aplicado a {len(cols)} columnas | Train {st.session_state.train_df.shape} | Test {st.session_state.test_df.shape}")
                    st.caption(f"Guardado en: {oh_path.resolve()}")
                    st.code(oh_path.read_text(encoding='utf-8')[:2000], language='text')
        with tab4:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: enriquecer el dataset con variables derivadas (p.ej., HHI, LQ, IPG) y dummies adicionales alineadas al problema.
            - Comportamiento esperado: aparecen nuevas columnas; se guarda un reporte de ingeniería de características.
            - Botón:
              - Generar features (HHI, LQ, IPG, dummies): calcula features y actualiza train/test; guarda feature_engineering_report.txt.
            Notas: si repites el paso, verifica que no dupliques columnas; en caso de dudas, rehaz desde el split.
            """)
            if st.session_state.get("train_df") is None:
                st.info("Ejecuta pasos previos")
            else:
                if st.button("Generar features (HHI, LQ, IPG, dummies)", use_container_width=True):
                    train_fe, test_fe, rep = engineer_features(st.session_state.train_df, st.session_state.test_df)
                    st.session_state.train_df = train_fe
                    st.session_state.test_df = test_fe
                    fe_path = Path("data/processed/feature_engineering_report.txt")
                    fe_path.write_text(rep, encoding="utf-8")
                    st.success("Features generados")
                    st.caption(f"Guardado en: {fe_path.resolve()}")
                    st.code(rep[:5000])
        with tab5:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: diagnosticar multicolinealidad y relaciones entre variables antes de seleccionar features.
            - Comportamiento esperado: se generan archivos con matriz de correlación y VIF en data/processed y se muestra un preview.
            - Botón:
              - Calcular correlación y VIF: computa y guarda correlation_matrix.csv y vif_scores.csv; ayuda a decidir eliminación de variables redundantes.
            Notas: trabajar sobre train para evitar sesgos.
            """)
            if st.session_state.get("train_df") is None:
                st.info("Ejecuta pasos previos")
            else:
                if st.button("Calcular correlación y VIF", use_container_width=True):
                    corr = correlation_matrix(st.session_state.train_df, sample_rows=20000, max_cols=200)
                    vif = compute_vif(st.session_state.train_df, sample_rows=20000, corr_thresh=0.90, var_thresh=1e-5, max_cols=200)
                    st.success("Guardado: correlation_matrix.csv, vif_scores.csv en data/processed")
                    corr_path = Path("data/processed/correlation_matrix.csv")
                    vif_path = Path("data/processed/vif_scores.csv")
                    st.caption(f"Correlación: {corr_path.resolve()}")
                    if corr_path.exists():
                        import pandas as _pd
                        st.dataframe(_pd.read_csv(corr_path).head(20), use_container_width=True)
                    st.caption(f"VIF: {vif_path.resolve()}")
                    st.dataframe(vif.head(20), use_container_width=True)
        with tab6:
            st.markdown("""
            Descripción de la sección
            - Razón de existir: reducir dimensionalidad y persistir los conjuntos finales para modelado.
            - Comportamiento esperado: se define una lista de features en memoria y se guardan X_train/X_test procesados en disco.
            - Botones:
              - Seleccionar features: ejecuta un selector (según tarea) y guarda la lista en sesión para filtrar columnas.
              - Guardar datasets: escribe X_train_engineered.csv y X_test_engineered.csv en data/processed (sin columnas objetivo);
                si hay selección previa, sólo guarda esas columnas.
            Notas: estos archivos son la entrada de la fase de entrenamiento.
            """)
            if st.session_state.get("train_df") is None:
                st.info("Ejecuta pasos previos")
            else:
                colA, colB = st.columns(2)
                if colA.button("Seleccionar features", use_container_width=True):
                    task = "classification" if config.TARGET_CLASSIFICATION in st.session_state.train_df.columns else "regression"
                    y_col = config.TARGET_CLASSIFICATION if task == "classification" else config.TARGET_REGRESSION
                    y_train = st.session_state.train_df[y_col] if y_col in st.session_state.train_df.columns else pd.Series([0]*len(st.session_state.train_df))
                    X_train = st.session_state.train_df.drop(columns=[c for c in [config.TARGET_CLASSIFICATION, config.TARGET_REGRESSION] if c in st.session_state.train_df.columns])
                    selected = select_features(X_train, y_train, task=task)
                    st.session_state.selected_features = selected
                    st.success(f"{len(selected)} features seleccionadas")
                    st.write(selected[:20])
                    sel_path = Path("data/processed/selected_features.txt")
                    if sel_path.exists():
                        st.caption(f"Guardado en: {sel_path.resolve()}")
                        st.code(sel_path.read_text(encoding='utf-8')[:2000], language='text')
                if colB.button("Guardar datasets", use_container_width=True):
                    X_train = st.session_state.train_df.drop(columns=[c for c in [config.TARGET_CLASSIFICATION, config.TARGET_REGRESSION] if c in st.session_state.train_df.columns])
                    X_test = st.session_state.test_df.drop(columns=[c for c in [config.TARGET_CLASSIFICATION, config.TARGET_REGRESSION] if c in st.session_state.test_df.columns])
                    if st.session_state.get("selected_features"):
                        X_train = X_train[X_train.columns.intersection(st.session_state.selected_features)]
                        X_test = X_test[X_test.columns.intersection(st.session_state.selected_features)]
                    save_datasets(X_train, X_test)
                    xtr = Path("data/processed/X_train_engineered.csv")
                    xte = Path("data/processed/X_test_engineered.csv")
                    st.success("Guardado: X_train_engineered.csv, X_test_engineered.csv")
                    st.caption(f"Train: {xtr.resolve()}")
                    st.caption(f"Test: {xte.resolve()}")
                    try:
                        st.dataframe(pd.read_csv(xtr).head(20), use_container_width=True)
                        st.dataframe(pd.read_csv(xte).head(20), use_container_width=True)
                    except Exception as _e:
                        st.warning(f"No se pudieron previsualizar los datasets: {_e}")

# ============ INFORMES ============
elif fase == "Informes":
    st.header("Informes")
    docs_dir = Path("docs")
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        st.info("No se encontraron archivos .md en docs")
    else:
        tabs = st.tabs([p.name for p in md_files])
        for tab, md in zip(tabs, md_files):
            with tab:
                st.caption(f"Ruta: {md.resolve()}")
                try:
                    st.markdown(md.read_text(encoding='utf-8'))
                except Exception as e:
                    st.error(f"Error leyendo {md.name}: {e}")

st.markdown("---")
