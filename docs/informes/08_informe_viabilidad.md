# Informe de Viabilidad: Despliegue de Modelos Predictivos

**Autores:**

- Arnoldo Aguirre  
- Juan Cerpa  

---

## 1.0 Introducción y Propósito del Informe

Este documento presenta una evaluación exhaustiva sobre la viabilidad técnica y operativa de dos modelos predictivos de alto rendimiento. Las soluciones, una de clasificación y otra de regresión, han superado con éxito la fase de modelado y evaluación, demostrando una precisión y robustez notables.

El propósito de este informe es proporcionar a la dirección una evaluación objetiva y fundamentada para tomar una decisión informada sobre el avance hacia la siguiente fase del proyecto: el desarrollo de una API de servicio y una interfaz de usuario (UI) para la puesta en producción de los modelos.

A través de este análisis, se validará si los activos de modelado cumplen con los requisitos técnicos, operacionales y estratégicos para su despliegue en un entorno real.

---

## 2.0 Activos de Modelado: Resumen de Soluciones Desarrolladas

Comprender los activos de modelado finales es un paso estratégico fundamental. Los dos modelos que se detallan a continuación representan la culminación del ciclo de desarrollo y son el objeto central de este análisis de viabilidad.

Ambos han sido seleccionados como los de mayor rendimiento tras un riguroso proceso de entrenamiento y validación comparativa que incluyó múltiples algoritmos (como Regresión Logística y Gradient Boosting), destacando en todas las métricas de éxito definidas.

### 2.1 Modelo de Clasificación: Predicción de Modalidad del Programa

Este modelo está diseñado para identificar si un programa académico se impartirá en modalidad presencial o no presencial.

| Característica              | Descripción                                  |
|----------------------------|----------------------------------------------|
| **Objetivo del Modelo**    | Predecir si un programa es Presencial o No Presencial. |
| **Algoritmo Seleccionado** | Random Forest                                |
| **Métrica Principal de Éxito** | Accuracy (Exactitud)                     |
| **Resultado Clave Obtenido** | 98.41%                                    |

### 2.2 Modelo de Regresión: Predicción de Edad Promedio

Este modelo está diseñado para estimar con una precisión excepcional la edad promedio de los estudiantes que cursarán un determinado programa académico, un dato clave para la planificación de recursos y servicios.

| Característica              | Descripción                                                |
|----------------------------|------------------------------------------------------------|
| **Objetivo del Modelo**    | Predecir la edad promedio de los estudiantes por programa. |
| **Algoritmo Seleccionado** | Random Forest                                              |
| **Métrica Principal de Éxito** | R² (Coeficiente de determinación)                     |
| **Resultado Clave Obtenido** | 0.9985 (explica el 99.85% de la varianza)              |

Habiendo definido los modelos seleccionados, la siguiente sección valida rigurosamente su rendimiento técnico y su preparación para un entorno de producción.

---

## 3.0 Evaluación de la Viabilidad Técnica: Rendimiento y Robustez

Esta sección constituye la validación fundamental de la calidad y fiabilidad de los modelos. Un rendimiento excepcional y una capacidad de generalización comprobada son requisitos indispensables para cualquier despliegue en un entorno productivo, ya que garantizan que las predicciones serán precisas y estables a lo largo del tiempo.

### 3.1 Superación de Métricas de Éxito

Los modelos no solo cumplieron, sino que superaron holgadamente todos los objetivos de rendimiento preestablecidos al inicio del proyecto, lo que confirma su alta calidad técnica.

**Comparativa: Métrica de Éxito vs. Resultado del Modelo**

| Métrica                | Objetivo Definido | Resultado Obtenido | Estado    |
|------------------------|------------------:|-------------------:|----------:|
| Clasificación - Accuracy | > 85%          | 98.41%             | Superado  |
| Clasificación - F1-Score | > 0.75        | 0.9821             | Superado  |
| Regresión - R²         | > 0.70           | 0.9985             | Superado  |
| Regresión - MAE        | < 2.0 años       | 0.0963 años        | Superado  |

### 3.2 Análisis de Generalización y Ausencia de Sobreajuste (*Overfitting*)

Un aspecto crítico de la validación fue asegurar que los modelos no solo funcionaran bien con los datos utilizados para su entrenamiento, sino que también pudieran generalizar su rendimiento a datos nuevos y futuros. Esta capacidad es esencial para garantizar su utilidad y fiabilidad a largo plazo en un entorno operativo.

El análisis comparativo entre el rendimiento del modelo en los datos de entrenamiento (validado mediante una técnica robusta de *cross-validation*) y su rendimiento en un conjunto de datos de prueba completamente nuevo confirma la excelente capacidad de generalización de ambos modelos.

- **Modelo de Clasificación:**  
  La diferencia de rendimiento entre la validación y la prueba fue de apenas **+0.0004**, una variación prácticamente nula que indica una estabilidad excepcional.

- **Modelo de Regresión:**  
  De manera similar, la diferencia fue de **+0.0007**, lo que reafirma su consistencia en datos no vistos.

**Conclusión:**  
Los resultados demuestran de forma concluyente que ambos modelos poseen una excelente generalización y no presentan ninguna evidencia de sobreajuste (*overfitting*).

La probada solidez técnica de los modelos es un pilar fundamental de su viabilidad. El siguiente análisis se centrará en su valor práctico y operacional.

---

## 4.0 Análisis de la Viabilidad Operacional: Impacto y Consideraciones de Negocio

La viabilidad de un modelo predictivo no depende únicamente de su precisión técnica, sino también de su capacidad para generar valor tangible y conocimientos accionables. Esta sección analiza los factores predictivos clave que los modelos han identificado y traduce estos hallazgos en implicaciones estratégicas para la toma de decisiones.

### 4.1 Implicaciones para la Predicción de Modalidad

El análisis de interpretabilidad del modelo de clasificación revela un hallazgo de gran importancia estratégica: la variable **JORNADA** es el factor predictivo dominante, explicando el **57.97%** de la capacidad del modelo para determinar la modalidad de un programa.

Las implicaciones de negocio de este hallazgo son claras y directas:

- **Relación Intrínseca:**  
  La modalidad de un programa (Presencial/No Presencial) está intrínsecamente ligada a su estructura de horarios (diurna, vespertina, a distancia, etc.).

- **Planificación Estratégica:**  
  Este hallazgo implica que cualquier estrategia para modificar la oferta de modalidades es inseparable de una revisión directa de la estructura y oferta de las jornadas de los programas.

### 4.2 Implicaciones para la Predicción de Edad Promedio

En el modelo de regresión, el análisis es aún más concluyente. Las variables **PROMEDIO EDAD HOMBRE (58.78%)** y **PROMEDIO EDAD MUJER (37.18%)** explican conjuntamente el **95.96%** de la varianza en la edad promedio de los estudiantes.

La principal implicación de negocio de este hallazgo es la alta fiabilidad de la predicción:

- **Relación Determinística:**  
  La edad promedio de los estudiantes de un programa es casi una función determinística de su composición demográfica por género.  
  Esto representa una ventaja significativa, ya que la alta fiabilidad del modelo reduce el riesgo operacional al no depender de variables *proxy* complejas o potencialmente inestables.

- **Palanca de Acción – Políticas de Admisión:**  
  Para influir en la edad promedio, la palanca de acción es clara: las políticas de admisión.

Los modelos no solo son técnicamente precisos, sino que también proporcionan una visión clara y procesable de las dinámicas subyacentes, lo que confirma su viabilidad operacional y justifica la planificación de su despliegue.

---

## 5.0 Hoja de Ruta Propuesta para la Puesta en Producción

Con la viabilidad técnica y operacional confirmada, se propone el siguiente plan de acción estratégico y escalonado para llevar los modelos validados desde el entorno de desarrollo a un sistema productivo plenamente funcional, capaz de entregar valor de forma continua.

### 5.1 Corto Plazo (1–2 semanas)

- **Desarrollo de API REST:**  
  Crear los endpoints de servicio `/predict/modalidad` y `/predict/edad` para permitir que otras aplicaciones consuman las predicciones de los modelos de forma programática.

- **Construcción de UI Interactiva:**  
  Desarrollar un panel de control (*dashboard*) que incluya:
  - Un formulario para realizar predicciones manuales.  
  - Visualizaciones de la importancia de las variables.  
  Esto facilita el uso por parte de usuarios no técnicos.

- **Implementación de Monitoreo Básico:**  
  Establecer sistemas para:
  - El seguimiento del rendimiento del modelo.  
  - La detección de derivas en los datos (*data drift*).  
  - La configuración de un sistema de alertas.

### 5.2 Mediano Plazo (1–3 meses)

- **Mejoras de Datos:**  
  Planificar la recolección de datos del próximo ciclo (2025) para:
  - Reentrenar los modelos.  
  - Evaluar cualquier degradación de su rendimiento.

- **Ingeniería de Características Avanzada:**  
  Explorar la creación de nuevas variables a partir de:
  - Interacciones entre variables.  
  - Características temporales.  
  - Agrupaciones geográficas.  

- **Ajuste Fino de Hiperparámetros:**  
  Ejecutar procesos de optimización sistemática para:
  - Exprimir el rendimiento máximo de los modelos.  
  - Asegurar la eficiencia predictiva.

### 5.3 Largo Plazo (3–12 meses)

- **Exploración de *Deep Learning*:**  
  Investigar el uso de redes neuronales para datos estructurados como una posible evolución de los modelos actuales.

- **Explicabilidad Avanzada (XAI):**  
  Implementar técnicas como:
  - SHAP, para ofrecer justificaciones a nivel de predicción individual.  
  - Otras herramientas que aumenten la transparencia y la confianza de los usuarios finales en la herramienta.

- **Infraestructura de Producción Robusta:**  
  Desarrollar un marco para:
  - Pruebas A/B.  
  - Versionado de modelos.  
  - Sistema de reentrenamiento automatizado.  
  - Garantizar la sostenibilidad a largo plazo.

Esta hoja de ruta proporciona un camino claro hacia la implementación, pero su éxito depende de una gestión proactiva de los riesgos potenciales, como se detalla a continuación.

---

## 6.0 Análisis de Riesgos y Estrategias de Mitigación

Una gestión de riesgos proactiva es un pilar fundamental para asegurar la fiabilidad, la relevancia y el éxito a largo plazo de las soluciones de inteligencia artificial desplegadas. A continuación, se identifican los riesgos clave y se proponen estrategias concretas para su mitigación.

| Riesgo Identificado                                | Probabilidad | Impacto | Estrategia de Mitigación Propuesta                                                                                                       |
|----------------------------------------------------|-------------:|--------:|-------------------------------------------------------------------------------------------------------------------------------------------|
| Data Drift (Deriva de datos con nuevos datos 2025) | Alta         | Alto    | Implementar un sistema de monitoreo automático para detectar cambios en la distribución de los datos de entrada y activar reentrenamiento. |
| Desbalance Extremo Futuro en Clases                | Media        | Alto    | Utilizar estrategias de remuestreo (*resampling*) o ajustar los pesos de las clases durante el reentrenamiento.                          |
| Cambios en la Importancia de Variables             | Media        | Medio   | Validar periódicamente la importancia de las variables con nuevos datos y documentar cambios significativos.                             |
| Cambios en Políticas Educacionales                 | Baja         | Alto    | Mantener comunicación regular con stakeholders del Ministerio para anticipar cambios regulatorios o estratégicos.                        |

La existencia de estas estrategias de mitigación proactivas demuestra que el perfil de riesgo del proyecto es aceptable y está bajo control, validando la recomendación de avanzar.

---

## 7.0 Veredicto Final y Recomendación

Tras un análisis integral de los aspectos técnicos, operacionales y estratégicos, este informe recapitula los hallazgos clave que fundamentan la decisión de despliegue.

El análisis ha demostrado de manera concluyente los siguientes puntos:

1. **Rendimiento Excepcional:**  
   Los modelos Random Forest seleccionados superaron todas las métricas de éxito predefinidas, alcanzando niveles de precisión y R² cercanos a la perfección.

2. **Robustez y Fiabilidad:**  
   Se ha confirmado la ausencia total de sobreajuste, garantizando que los modelos generalizan su rendimiento de manera excelente a datos no vistos.

3. **Insights Accionables:**  
   Los modelos no solo predicen con exactitud, sino que también revelan los factores clave que impulsan los resultados, proporcionando *insights* de negocio claros y accionables.

4. **Hoja de Ruta Clara:**  
   Existe un plan de implementación detallado y por fases que traza un camino viable desde el desarrollo hasta una solución productiva robusta.

5. **Riesgos Gestionables:**  
   Los riesgos operacionales inherentes al ciclo de vida de los modelos han sido identificados y cuentan con estrategias de mitigación definidas.

**Conclusión y Recomendación:**

Basado en la evaluación exhaustiva presentada en este informe, se determina que los modelos de clasificación y regresión desarrollados son **altamente viables para su despliegue en un entorno de producción**, donde se espera que generen un valor tangible para la planificación estratégica del Ministerio.
