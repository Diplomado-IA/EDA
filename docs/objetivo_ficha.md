# Ficha de Objetivos del Proyecto ML

## Problema
Optimizar la gestión académica y recursos mediante modelos predictivos que anticipen la modalidad (Presencial vs No presencial) y la edad promedio de titulación por programa en Educación Superior Chile (2007–2024).

## Variables Objetivo (Y)
- Clasificación: MODALIDAD_BIN (Presencial=1, No presencial=0)
- Regresión: PROMEDIO_EDAD_PROGRAMA (valor numérico continuo)

## Variables Explicativas (X)
- Área CINE / Sub-área CINE
- Región / Comuna
- Jornada
- Duración total (semestres)
- Tamaño del programa (matrícula / titulaciones)
- Año
- Institución

## Métricas de Éxito
- Clasificación: AUC-PR (robusta ante desbalance), F1-macro (peso igual a ambas clases)
- Regresión: MAE (interpretabilidad en años), RMSE (penaliza errores grandes)

## Riesgos Identificados
- Desbalance de clases (Presencial >> No presencial)
- Alta nulidad o inconsistencias en rangos etarios
- Drift temporal post-2020 (cambios pandemia)

## Decisiones Iniciales
- Usar estratificación por MODALIDAD_BIN en splits
- Evaluar uso de class_weight y ajuste de umbral operativo según Recall clase minoritaria
- Convertir edades y duraciones a tipos numéricos consistentes

## Criterios de Aceptación Fase 1
- Este documento existe y contiene Y, X, métricas y riesgos
- Config actualizado con nombres de targets y métricas
- Riesgos documentados para guiar EDA y preprocesamiento

## Verificación
```bash
ls docs/objetivo_ficha.md && grep -q 'MODALIDAD_BIN' docs/objetivo_ficha.md && echo OK_FASE1

python -c "from config.config import Config; print(Config.TARGET_CLASSIFICATION, Config.TARGET_REGRESSION, Config.METRICS)"
```
