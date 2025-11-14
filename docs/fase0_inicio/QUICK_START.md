# ⚡ Quick Start - Ejecutar Proyecto

## 🚀 Inicio Rápido (5 minutos)

### Opción 1: CLI (Recomendado para automatización)

```bash
# Posicionarse en el directorio
cd /home/anaguirv/ia_diplomado/EDA

# Activar virtual environment
source venv/bin/activate

# Ver configuración
python main.py --mode config

# Ejecutar EDA
python main.py --mode eda

# Ver resultados
ls outputs/eda/
```

**Salida esperada:** 4 gráficos PNG en `outputs/eda/`

---

### Opción 2: UI (Recomendado para demos)

```bash
cd /home/anaguirv/ia_diplomado/EDA
source venv/bin/activate

streamlit run ui/app.py
```

**Luego:** Abre http://localhost:8501

---

### Opción 3: Python (Recomendado para desarrollo)

```python
from src.pipeline import MLPipeline

# Crear pipeline
pipeline = MLPipeline()

# Ejecutar EDA
pipeline.run_eda_only()

# Ver resultados en outputs/eda/
```

---

## 📊 Archivos Clave

```
src/
├── config.py              ← Configuración centralizada
├── pipeline.py            ← Orquestador (USAR ESTE)
├── data/cleaner.py        ← Carga y limpieza
└── visualization/eda.py   ← Visualizaciones

main.py                     ← CLI (usar con: python main.py --mode eda)
ui/app.py                   ← UI (usar con: streamlit run ui/app.py)

outputs/
└── eda/                    ← Gráficos generados aquí
```

---

## ✅ Validación Rápida

```bash
# Test 1: CLI Config
python main.py --mode config
# ✓ Debe mostrar JSON con configuración

# Test 2: CLI EDA
python main.py --mode eda
# ✓ Debe generar 4 PNG en outputs/eda/

# Test 3: UI
streamlit run ui/app.py
# ✓ Debe abrir navegador en http://localhost:8501
```

---

## 🔗 Documentos de Referencia

- **ARQUITECTURA_MODULAR.md** - Diseño completo
- **GUIA_EJECUCION_MODULAR.md** - Guía detallada
- **ONBOARDING_EQUIPO.md** - Para colegas
- **SOLUCION_ERRORES_EDA.md** - Errores y soluciones

---

## 💡 Comandos Útiles

```bash
# Ver ayuda
python main.py --help

# EDA con verbose
python main.py --mode eda --verbose

# EDA con paths personalizados
python main.py --mode eda --output mi_carpeta/

# Entrenar modelos (próxima fase)
python main.py --mode train
```

---

**¡Listo para usar!** 🎉

Próximo paso: `python main.py --mode eda`
