#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                   🎓 INTERFAZ INTERACTIVA - EVALUACIÓN ML                      ║"
echo "║              Ejecuta paso a paso el proceso según rúbrica 03M5U2              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "ui/pipeline_executor.py" ]; then
    echo "❌ Error: No se encuentra ui/pipeline_executor.py"
    exit 1
fi

echo "✅ Proyecto encontrado"
echo ""

if [ -d "venv" ]; then
    echo "✅ Virtual environment encontrado"
    echo "Activando venv..."
    source venv/bin/activate
    echo "✅ venv activado"
else
    echo "⚠️  No se encontró venv"
fi

echo ""
echo "📦 Verificando dependencias..."
pip install -q streamlit pandas numpy matplotlib seaborn 2>/dev/null

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                                ║"
echo "║  🚀 Iniciando interfaz interactiva...                                          ║"
echo "║                                                                                ║"
echo "║  📍 URL: http://localhost:8501                                                 ║"
echo "║  🎯 Funcionalidad: Ejecutar paso a paso (Similar a Jupyter Notebook)          ║"
echo "║                                                                                ║"
echo "║  📋 CÓMO USAR:                                                                 ║"
echo "║     1. Selecciona un paso (1-8) del menú lateral                               ║"
echo "║     2. Haz clic en '▶️ EJECUTAR PASO'                                           ║"
echo "║     3. Observa resultados y métricas                                          ║"
echo "║     4. Navega a siguiente paso                                                ║"
echo "║     5. Completa los 8 pasos de la rúbrica                                      ║"
echo "║                                                                                ║"
echo "║  ✅ Para cerrar: CTRL + C                                                      ║"
echo "║                                                                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

streamlit run ui/pipeline_executor.py
