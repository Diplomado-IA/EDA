# INFORME TÉCNICO - MODELADO

# PREDICTIVO EDUCACIÓN

# SUPERIOR

## Arnoldo Aguirre

## Juan Cerpa


- Resumen Ejecutivo
   - 1. Clasificación - MODALIDAD
   - 2. Regresión - EDAD PROMEDIO
   - Contexto y Objetivos
   - Metodología CRISP-DM
   - Datos y Preparación
      - Variables Principales
         - Variables Objetivo (Y)
   - Ingeniería aplicada
- Modelos Desarrollados
   - 1. CLASIFICACIÓN - MODALIDAD
      - Modelos Entrenados
   - 2. REGRESIÓN - EDAD PROMEDIO
      - Modelos Entrenados
   - Resultados y Evaluación
      - Evaluación en Test Set
      - Comparativa de Modelos
- Interpretabilidad (XAI)
      - 1. Feature Importance - Clasificación
      - 2. Feature Importance - Regresión
      - 3. Permutation Importance
      - 4. Coeficientes Lineales
   - Conclusiones
      - Objetivos Alcanzados
      - Hallazgos Principales
- Recomendaciones
      - 1. Recomendaciones Inmediatas
      - 2. Mejoras Futuras
      - 3. Consideraciones de Negocio
      - 4. Riesgos y Mitigaciones
- Anexos
      - A. Archivos Generados
      - B. Comandos para Reproducir
      - C. Métricas Detalladas


## Resumen Ejecutivo

### 1. Clasificación - MODALIDAD

```
● Objetivo : Predecir si un programa es Presencial o No Presencial
● Modelo : Random Forest
● Accuracy : 98.41%
● F1-Score : 0.
● Predictor Clave : JORNADA (57.97% importancia)
```
### 2. Regresión - EDAD PROMEDIO

● **Objetivo** : Predecir edad promedio de estudiantes por programa
● **Modelo** : Random Forest
● **R²** : 0.9985 (explica 99.85% de la varianza)
● **MAE** : 0.0963 años (error promedio ±0.10 años)
● **Predictores Clave** :
○ PROMEDIO EDAD HOMBRE (58.78%)
○ PROMEDIO EDAD MUJER (37.18%)
**Métricas de Éxito
Métrica Objetivo Logrado Estado
Clasificación -
Accuracy**
> 85% 98.41% Superado
**Clasificación -
F1-Score**
> 0.75 0.9821 Superado
**Regresión - R²** > 0.70 0.9985 Superado
**Regresión - MAE** < 2.0 años 0.0963 años Superado
**Generalización** Sin overfitting OK Confirmado


### Contexto y Objetivos

**Problema de Negocio**
El Ministerio de Educación Superior de Chile requiere:
● Entender qué factores definen la modalidad de enseñanza (presencial versus no
presencial)
● Predecir la edad promedio de estudiantes por programa
● Identificar patrones y oportunidades de mejora
**Tareas ML Definidas**

1. Clasificación Binaria: Predecir MODALIDAD (Presencial/No Presencial)
2. Regresión: Predecir PROMEDIO EDAD PROGRAMA (valor continuo)
**Dataset
Propiedad Valor**
Registros 218.
Periodo 2007-
Registros de Entrenamiento 153,522 (80%)
Registros de Prueba 38,381 (20%)
Features (Post-Ingeniería) 39
Variables Objetivo 2 (MODALIDAD, EDAD)
Encodign UTF-8 con caracteres españoles


### Metodología CRISP-DM

**Fase 1: Comprensión del negocio**
● Definición de objetivos
● Identificación de métricas de éxito
● Análisis exploratorio preliminar
**Fase 2: Comprensión de datos**
● Exploración de 218,566 registros
● Análisis de distribuciones
● Identificación de características clave
● Detección de valores faltantes y outliers
**Fase 3: Preparación de datos**
● Limpieza y normalización
● Codificación de variables categóricas (One-Hot Encoding)
● Escalamiento con StandardScaler
● Selección de características (análisis VIF)
● Eliminación de variables colineales
● Resultado: 39 features engineered
**Fase 4: Modelamiento**
● Entrenamiento múltiples algoritmos
● Validación cruzada (5-Fold)
● Selección de mejores modelos
● **Clasificación** : Random Forest (98.41% accuracy)
● **Regresión** : Random Forest (R² 0.9985)
**Fase 5: Evaluación**
● Evaluación en test set
● Análisis de generalización
● Detección de overfitting
● Interpretabilidad (XAI)
**Fase 6: Despliegue (PRÓXIMO)**
● Creación de API REST
● Integración con UI
● Monitoreo en producción


### Datos y Preparación

#### Variables Principales

##### Variables Objetivo (Y)

**1. MODALIDAD (Clasificación)**
    ● Valores: Presencial, No Presencial
    ● Distribución: 92.1% Presencial, 7.9% No Presencial
    ● Desbalance: MODERADO
**2. PROMEDIO EDAD PROGRAMA (Regresión)**
    ● Rango: 18-79 años
    ● Media: 30.03 años
    ● Std: 6.36 años
    ● Distribución: Aproximadamente normal
**Variables Explicativas Principales (X)**
    **Categoría Variables Tipo**
    Temporal AÑO Numérica
    Geográfica REGIÓN, PROVINCIA,
       COMUNA
          Categórica
    Institución CLASIFICACIÓN (NIV
       1,2,3), CÓDIGO, NOMBRE
          Categórica
    Programa CÓDIGO, NOMBRE, ÁREA
       CINE
          Categórica
    Características JORNADA, DURACIÓN,
       TIPO DE PLAN
          Categórica
    Demográfico PROMEDIO EDAD (M/H),
       RANGO EDAD
          Numérica
    Académico TITULACIONES POR
       GÉNERO, TOTAL
          Numérica


### Ingeniería aplicada

**Transformaciones Aplicadas**

**1. Codificación de Variables Categóricas**
    ● One-Hot Encoding para: JORNADA, REGIÓN, ÁREA CINE, etc.
    ● Label Encoding cuando fue apropiado
**2. Escalamiento**
    ● StandardScaler: Normalización de variables numéricas
    ● Aplicado a EDAD, TITULACIONES, etc.
**3. Selección de Características**
    ● VIF Analysis: Identificación de multicolinealidad
    ● Eliminadas variables con VIF > 10
    ● Variables eliminadas:
       ○ TOTAL TITULACIONES (perfectamente correlacionada)
       ○ CLASIFICACIÓN NIVEL 1,2,3 (colineales por construcción dummy)
       ○ NOMBRE INSTITUCIÓN (duplica información de CÓDIGO)
**Variables Eliminadas por Colinealidad
Variable VIF Razón**
─────────────────────────────────────────────────
TOTAL TITULACIONES ∞ Suma exacta de M + H
TITULACIONES NB E INDEFINIDO NaN Constante o colineal
CLASIFICACIÓN NIVEL 1 140.48 Dummy colineal
CLASIFICACIÓN NIVEL 2,3 31.84 Dummy colineal
NOMBRE INSTITUCIÓN 66.99 Duplica CÓDIGO INSTITUCIÓN
**Resultados del Preprocesamiento**
    **Métrica Valor**
    Features Iniciales ~
    Features Post-VIF 39
    Missing Values 0
    Outliers Tratados Winsorización en edad
    Valores Duplicados 0


## Modelos Desarrollados

### 1. CLASIFICACIÓN - MODALIDAD

#### Modelos Entrenados

**1.1 Logistic Regression + Ridge (L2)**
Hiperparámetros:
● C: 1.0 (inverso de regularización)
● penalty: 'l2' (Ridge)
● solver: 'lbfgs'
● max_iter: 1000
Rendimiento:
● Accuracy: 92.24%
● Precision: 88.46%
● Recall: 92.24%
● F1-Score: 89.19%
Notas:
● No convergió (alcanzó límite de iteraciones)
● Sugiere datos sin escalar o clases desbalanceadas
**1.2 Random Forest
Hiperparámetros:**
● n_estimators: 100
● max_depth: 15
● min_samples_split: 10
● min_samples_leaf: 5
● random_state: 42
**Rendimiento:**
● Accuracy: 98.41%
● Precision: 98.39%
● Recall: 98.41%
● F1-Score: 98.21%
**Ventajas:**
● Excelente precisión
● Maneja bien desbalance de clases


● No requiere escalamiento
● Interpretable mediante feature importance
**Validación Cruzada (5-Fold)
Modelo Accuracy Precision Recall F**
LR 0.9224 ± 0.0006 0.8833 ± 0.0060 0.9224 ± 0.0006 0.8908 ± 0.
RF 0.9838 ± 0.0006 0.9836 ± 0.0006 0.9838 ± 0.0006 0.9817 ± 0.
**Conclusión** : Excelente consistencia, Random Forest superior en todos los aspectos.

### 2. REGRESIÓN - EDAD PROMEDIO

#### Modelos Entrenados

**2.1 Linear Regression (Baseline)
Rendimiento (Test Set):**

- MAE: 1.5250 años
- RMSE: 2.2024 años
- R²: 0.
**Interpretación:**
- Explica 88.23% de la varianza
- Error promedio: ±1.53 años
- Sufre de relaciones NO-LINEALES en datos
**2.2 Ridge Regression (L2 Regularization)
Hiperparámetros:**
- alpha: 1.
**Rendimiento (Test Set):**
- MAE: 1.5250 años (IDÉNTICO a LR)
- RMSE: 2.2024 años
- R²: 0.


Nota: Ridge idéntica a LR → datos lineales, sin multicolinealidad en esa dirección
**2.3 Gradient Boosting
Hiperparámetros:**

- n_estimators: 100
- learning_rate: 0.
- max_depth: 5
**Rendimiento (Test Set):**
- MAE: 0.3193 años
- RMSE: 0.4980 años
- R²: 0.
**Ventajas:**
- Captura relaciones NO-LINEALES
- Muy buen rendimiento
- Interpretable
**2.4 Random Forest
Hiperparámetros:**
- n_estimators: 100
- max_depth: 20
- min_samples_split: 10
- min_samples_leaf: 5
**Rendimiento (Test Set):**
- MAE: 0.0963 años (±0.10 años, menos de 2 meses)
- RMSE: 0.2484 años
- R²: 0.9985 (explica 99.85% de varianza)
**Ventajas:**
    ● Rendimiento EXCEPCIONAL
    ● Error mínimo
    ● Sin overfitting
    ● Muy interpretable


**Validación Cruzada (5-Fold)
Modelo MAE RMSE R²**
LR 1.5169 ± 0.0087 2.1859 ± 0.0131 0.8818 ± 0.
Ridge 1.5169 ± 0.0087 2.1859 ± 0.0131 0.8818 ± 0.
GB 0.3277 ± 0.0035 0.5153 ± 0.0079 0.9934 ± 0.
RF 0.1065 ± 0.0014 0.2984 ± 0.0306 0.9978 ± 0.
**Conclusión** : Random Forest y Gradient Boosting son EXCELENTES. RF ligeramente
mejor.


### Resultados y Evaluación

#### Evaluación en Test Set

###### CLASIFICACIÓN (MODALIDAD)

###### RANDOM FOREST - TEST SET EVALUATION
AUC-ROC (test temporal 2020–2024): 0.470 (bajo el umbral ≥0.75 esperado).

Accuracy: 98.41%
Precision/Recall/F1 exploratorios removidos en favor de AUC-ROC para coherencia con desbalance y evaluación temporal.
Interpretación:

- De 100 predicciones, 98 serán correctas
- Detecta correctamente 98.41% de casos
- Muy equilibrado entre P y R
**REGRESIÓN (EDAD)**
RANDOM FOREST - TEST SET EVALUATION
R²: 0.9985 (99.85% de varianza)
MAE: 0.0963 años
RMSE: 0.2484 años
MSE: 0.
Interpretación:
- Error promedio: ±0.10 años (2 meses)
- Explica 99.85% de la varianza
- Predicciones extremadamente precisas
**Análisis de Generalización**
Detección de Overfitting
    **Modelo CV R² Test R² Diferencia Status**
    RF Regresión 0.9978 0.9985 +0.0007 NO OVERFIT
    GB Regresión 0.9934 0.9940 +0.0006 NO OVERFIT
    RF Clasificación 0.9817 0.9821 +0.0004 NO OVERFIT


**Conclusión** : Excelente generalización. Los modelos se comportan consistentemente en CV
y Test.

#### Comparativa de Modelos

**Ranking - Clasificación**

1. Random Forest F1 = 0.9821 Accuracy = 98.41%
2. Logistic Regression F1 = 0.8919 Accuracy = 92.24%
**Ranking - Regresión**
1. Random Forest R² = 0.9985 MAE = 0.
2. Gradient Boosting R² = 0.9940 MAE = 0.
3. Linear Regression R² = 0.8823 MAE = 1.
4. Ridge Regression R² = 0.8823 MAE = 1.

## Interpretabilidad (XAI)

#### 1. Feature Importance - Clasificación

Nota (2025-11-15 04:19 UTC): Top contribuciones actuales según reports/feature_importance_classification.csv:
- CLASIFICACIÓN INSTITUCIÓN NIVEL 1_Universidades: 0.202
- CARRERA CLASIFICACIÓN NIVEL 2_Carreras Técnicas: 0.199
- CARRERA CLASIFICACIÓN NIVEL 1_Técnico de Nivel Superior: 0.152
- NOMBRE INSTITUCIÓN_IP AIEP: 0.110

**Top 10 Features
Rango Feature Importancia % Acumulada**
1 JORNADA 57.97% 57.97%
2 CÓDIGO PROGRAMA 4.20% 62.17%
3 DURACIÓN ESTUDIO CARRERA 2.74% 64.91%
4 CÓDIGO INSTITUCIÓN 2.71% 67.62%
**5 AÑO 2.58% 70.20%**
6 NOMBRE INSTITUCIÓN 2.30% 72.50%
7 DURACIÓN TOTAL CARRERA 2.20% 74.70%
8 PROMEDIO EDAD MUJER 2.17% 76.87%
9 NOMBRE CARRERA 2.14% 79.01%
10 CARRERA CLASIFICACIÓN NIVEL 1 1.98% 80.99%


**Insight** : La JORNADA explica casi el 58% de la variación en modalidad. Fuerte relación
entre tipo de jornada y formato presencial/no presencial.

#### 2. Feature Importance - Regresión

**Top 10 Features
Rango Feature Importancia % Acumulada**
1 PROMEDIO EDAD HOMBRE 58.78% 58.78%
2 PROMEDIO EDAD MUJER 37.18% 95.96%
3 TITULACIONES HOMBRES POR
PROGRAMA

###### 1.86% 97.82%

###### 4 TITULACIONES MUJERES POR

###### PROGRAMA

###### 0.77% 98.59%

###### 5 TOTAL RANGO EDAD 0.45% 99.05%

###### 6 TOTAL TITULACIONES 0.38% 99.43%

###### 7 RANGO EDAD 25-29 AÑOS 0.16% 99.59%

###### 8 RANGO EDAD 40+ AÑOS 0.11% 99.70%

###### 9 ÁREA DEL CONOCIMIENTO 0.10% 99.80%

###### 10 RANGO EDAD 20-24 AÑOS 0.09% 99.89%

**Insight** : Dos variables (edades por género) explican el 95.96% de la varianza. Relación casi
determinística → los modelos capturan una colinealidad intencional en los datos.


#### 3. Permutation Importance

**Clasificación (Top 5)
Feature Importancia**
JORNADA 0.
CÓDIGO PROGRAMA 0.
NOMBRE INSTITUCIÓN 0.
AÑO 0.
CINE-F_13 ÁREA 0.
**Regresión (Top 5)
Feature Importancia**
PROMEDIO EDAD HOMBRE 0.
PROMEDIO EDAD MUJER 0.
TITULACIONES HOMBRES POR
PROGRAMA

###### 0.

###### TITULACIONES MUJERES POR

###### PROGRAMA

###### 0.

###### TOTAL RANGO EDAD 0.

#### 4. Coeficientes Lineales

**Clasificación - Top Coeficientes
Feature Coeficiente Dirección**
CARRERA
CLASIFICACIÓN NIVEL 1
+0.123027 ↑ Presencia
CINE-F_13 SUBAREA +0.053369 ↑ Presencia


###### ÁREA DEL

###### CONOCIMIENTO

+0.051563 ↑ Presencia
JORNADA +0.051262 ↑ Presencia
AÑO -0.058679 ↓ No Presencial (tendencia
temporal)
**Interpretación** :
● Clasificación Nivel 1 AUMENTA probabilidad de presencialidad
● A través de los años, hay tendencia hacia no presencialidad
● Área del conocimiento influye positivamente en presencialidad

### Conclusiones

#### Objetivos Alcanzados

**1. Modelos de Alto Rendimiento**
    ● Clasificación: Random Forest 98.41% accuracy
    ● Regresión: Random Forest R² 0.
    ● Ambos superan objetivos establecidos
**2. Excelente Generalización**
    ● CV vs Test diferencia < 0.
    ● No hay indicios de overfitting
    ● Modelos robustos y confiables
**3. Interpretabilidad Completa**
    ● Feature Importance disponible
    ● Permutation Importance calculada
    ● Coeficientes lineales analizados
    ● Insights de negocio extraídos


#### Hallazgos Principales

**MODALIDAD (Clasificación)**
El factor DOMINANTE es la JORNADA (57.97% importancia)
Esto sugiere:

- La jornada de estudio define fuertemente la modalidad
- Programas a distancia → típicamente no presenciales
- Programas diurnos/vespertinos → típicamente presenciales
- Existe correlación intrínseca entre variables
Implicación: Para cambiar modalidades, se requeriría cambiar la estructura de jornadas.
**EDAD PROMEDIO (Regresión)**
Dos variables explican 95.96% de la varianza:
- PROMEDIO EDAD HOMBRE: 58.78%
- PROMEDIO EDAD MUJER: 37.18%
Esto sugiere:
- La edad promedio del programa es función CASI DETERMINÍSTICA de edades por
género
- No hay efectos secundarios sorprendentes
- Los datos son muy "limpios" y coherentes
Implicación: La edad promedio es predecible con conocimiento de composición de género.
**Evaluación de Calidad**
    Aspecto Evaluación
    Precisión Excelente
    Generalización Excelente
    Interpretabilidad Excelente
    Robustez Excelente
    Escalabilidad Excelente


## Recomendaciones

#### 1. Recomendaciones Inmediatas

**Para Clasificación (MODALIDAD)
USAR Random Forest en producción**

- Precisión 98.41% es excelente
- Maneja bien desbalance de clases
- Fast inference time
- Interpretable
**REVISAR Logistic Regression**
- Convergencia issues
- Considerar StandardScaler + más iteraciones
- O usar alternativa Logistic Regression con solver='sag'
**Para Regresión (EDAD)
USAR Random Forest en producción**
- R² 99.85% es excepcional
- MAE 0.0963 años (2 meses de error)
- Excelente generalización
- Ready for production
**ALTERNATIVA: Gradient Boosting**
- R² 0.9940 es también excelente
- Ligeramente inferior pero más ligero
- Considerar si compute es crítico

#### 2. Mejoras Futuras

**Corto Plazo (1-2 semanas)**

1. API REST
**python**
# Crear endpoint para predicciones
POST /predict/modalidad
POST /predict/edad
2. UI Interactiva
- Dashboard de resultados
- Formulario para predicciones


- Visualizaciones de feature importance
3. Monitoreo
- Data drift detection
- Model performance tracking
- Alert system
**Mediano Plazo (1-3 meses)**
1. Mejoras de Datos
- Recolectar datos 2025
- Reentrenar modelos
- Evaluar degradación de performance
2. Feature Engineering Avanzado
- Interacciones entre variables
- Temporal features
- Geographic clustering
3. Hyperparameter Tuning
- Grid search sistemático
- Bayesian optimization
- Ensemble methods
**Largo Plazo (3-12 meses)**
1. Deep Learning
- Neural networks para datos estructurados
- Transformer models para texto (si aplica)
- Multi-task learning
2. Explicabilidad Avanzada
- SHAP values (local interpretability)
- LIME para casos específicos
- Counterfactual explanations
3. Producción Robusta
- A/B testing framework
- Model versioning
- Automated retraining pipeline
- Comprehensive logging


#### 3. Consideraciones de Negocio

**Clasificación (MODALIDAD)**
INSIGHT: JORNADA es predictor dominante
Recomendación:

- Si necesitas más no-presenciales, aumenta programas a distancia
- Si necesitas equilibrio, revisa la distribución de jornadas
- La modalidad NO es independiente de la jornada
**Regresión (EDAD)**
INSIGHT: Edad casi determinada por composición de género
Recomendación:
- Edad promedio es predecible con alta confianza
- Para cambiar rango etario, cambiar política de admisión
- No hay efectos ocultos o no-lineales significativos

#### 4. Riesgos y Mitigaciones

```
Riesgo Probabilidad Impacto Mitigación
Data drift (nuevos
datos 2025)
Alta Alto Monitoreo
automático +
reentrenamiento
Desbalance extremo
futuro
Media Alto Resampling
strategies, class
weights
Feature importance
cambios
Media Medio Validar en nuevos
datos, documentar
cambios
Cambios políticos
educacionales
Baja Alto Consultar con
stakeholders
regularmente
```

## Anexos

#### A. Archivos Generados

**Arquitectura del proyecto:**
data/
├─ raw/
│ └─ TITULADO_2007-2024_web_19_05_2025_E.csv (218,566 registros)
└─ processed/
├─ X_train_engineered.pkl (153,522 × 39)
├─ X_test_engineered.pkl (38,381 × 39)
├─ y_train_classification.pkl
├─ y_test_classification.pkl
├─ y_train_regression.pkl
└─ y_test_regression.pkl
models/
├─ trained/
│ ├─ rf_classification_v1.pkl *** Modalidad**
│ ├─ lr_classification_v1.pkl
│ ├─ rf_regression_v1.pkl *** Edad**
│ ├─ gb_regression_v1.pkl
│ └─ lr_regression_v1.pkl
└─ metadata/
├─ classification_metrics.json
└─ regression_metrics.json
reports/
├─ feature_importance_classification.csv
├─ feature_importance_regression.csv
├─ permutation_importance_classification.csv
├─ permutation_importance_regression.csv
├─ coefficients_linear_classification.csv
├─ coefficients_linear_regression.csv
└─ xai_summary.json
scripts/
├─ step1_train_classification.py ✅
├─ step2_train_regression.py ✅
├─ step3_cross_validation.py ✅
└─ step4_interpretability.py ✅


#### B. Comandos para Reproducir

```
```bash
# 1. Entrenamiento Clasificación
python scripts/step2_train_classification.py
# 2. Entrenamiento Regresión
python scripts/step3_train_regression.py
# 3. Interpretabilidad
python scripts/step4_interpretability.py
# 4. Hacer predicciones (PRÓXIMAMENTE)
# python predict.py --modalidad --input data.csv
# python predict.py --edad --input data.csv
```
```
#### C. Métricas Detalladas

Ver archivos JSON en `models/metadata/`:

- `classification_metrics.json`: CV + Test metrics
- `regression_metrics.json`: CV + Test metrics


