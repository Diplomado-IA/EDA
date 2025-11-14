## Página 1


1. OBJETIVO DE APRENDIZAJE.
Examinar los algoritmos de Machine Learning en el contexto de modelado de datos, considerando la resolución
de problemáticas, agrupación, normalización anomalías y clasificación, según necesidades de la empresa.
2. DESCRIPCIÓN.
Construir un modelo de Machine Learning utilizando lenguaje Python o R de acuerdo con los casos entregados por el
docente, buscando mejorar el proceso de decisiones basada en datos.
3. INSTRUCCIONES.
A partir del caso y la base de datos entregado por el docente, debe realizar las siguientes actividades:
Comprensión del Caso y Objetivos del Modelo
• Analizar el Caso: Leer y comprender el contexto del caso proporcionado por el docente para entender
los objetivos del proyecto y las expectativas de los resultados.
• Definir el Objetivo del Modelo: Identificar si el modelo busca hacer predicciones, clasificaciones,
segmentaciones, o identificar patrones específicos que mejoren la toma de decisiones.
Análisis Exploratorio de Datos (EDA)
• Inspeccionar el Dataset: Revisar la estructura de los datos (columnas, tipos de datos, valores faltantes,
duplicados) para familiarizarse con la base de datos.
• Análisis Descriptivo: Calcular estadísticas descriptivas para las variables relevantes (media, mediana,
desviación estándar) y realizar visualizaciones para identificar distribuciones y relaciones entre variables.
pág. 1

## Página 2

• Detección y Tratamiento de Valores Faltantes: Identificar valores nulos o ausentes en el dataset y decidir
cómo tratarlos (por ejemplo, imputación de valores, eliminación de filas).
• Identificación de Outliers: Detectar y analizar valores atípicos que puedan afectar el rendimiento del
modelo y decidir cómo manejarlos.
Preprocesamiento de Datos
• Normalización o Estandarización: Escalar las variables numéricas si es necesario (especialmente para
modelos sensibles a la escala, como regresiones o redes neuronales).
• Codificación de Variables Categóricas: Convertir variables categóricas en formato adecuado para el
modelo (por ejemplo, One-Hot Encoding en Python o model.matrix en R).
• División del Dataset: Separar el dataset en conjuntos de entrenamiento y prueba (y validación, si es
necesario) para evaluar el rendimiento del modelo.
Selección del Modelo de Machine Learning
• Identificar Algoritmos Candidatos: Seleccionar uno o más algoritmos adecuados para el tipo de
problema, como regresión lineal, árboles de decisión, SVM, redes neuronales, entre otros.
• Entrenamiento Inicial: Entrenar los modelos candidatos con los datos de entrenamiento para obtener
una primera evaluación del desempeño.
• Optimización de Hiperparámetros: Realizar ajustes en los parámetros del modelo (Grid Search o
Random Search) para mejorar su precisión y evitar problemas de sobreajuste (overfitting).
Evaluación del Modelo
• Evaluación en el Conjunto de Validación o Prueba: Utilizar el conjunto de prueba para medir el
rendimiento del modelo con las métricas seleccionadas.
• Comparación de Modelos: Si se probaron varios algoritmos, comparar los resultados para seleccionar el
mejor modelo con base en su rendimiento y ajuste al caso de negocio.
• Validación Cruzada (si es necesario): Implementar validación cruzada para verificar la robustez del
modelo y asegurarse de que generaliza bien.
Interpretación de Resultados y Toma de Decisiones
• Análisis de Importancia de Variables: Identificar las variables más relevantes en el modelo para entender
los factores que más impactan las predicciones.
• Generación de Insights: Interpretar los resultados del modelo en el contexto del caso entregado por el
docente para proporcionar recomendaciones basadas en los datos.
• Evaluación de Impacto en la Toma de Decisiones: Analizar cómo los resultados del modelo pueden
mejorar el proceso de toma de decisiones.
Documentación y Presentación
pág. 2

## Página 3

• Documentación del Proceso: Escribir un informe que explique cada fase, las decisiones tomadas, el
rendimiento del modelo y los resultados obtenidos.
• Visualización de Resultados: Crear visualizaciones que ayuden a comunicar los resultados de forma
clara, incluyendo gráficos de desempeño del modelo y representaciones de la importancia de las
variables.
• Presentación: Preparar una presentación o un reporte ejecutivo para mostrar al docente o audiencia,
destacando cómo el modelo puede mejorar la toma de decisiones basada en datos.
Implementación y Recomendaciones Finales
• Implementación del Modelo (si aplica): En algunos casos, el modelo puede ser implementado en un
entorno productivo o servir como base para futuras predicciones.
• Recomendaciones para la Empresa o Contexto del Caso: Ofrecer recomendaciones prácticas basadas
en el análisis del modelo, considerando cómo podrían aplicarse los hallazgos para mejorar el proceso de
decisión.
pág. 3

## Página 4

4. EVALUACIÓN.
NIVEL DESCRIPCIÓN PUNTAJE
Ha logrado el desempeño óptimo, cumpliendo todos los aspectos
O ÓPTIMO 6
exigidos en el desarrollo de la tarea.
Ha logrado un desempeño satisfactorio al desarrollar la tarea, sólo
S SATISFACTORIO 4
debe atender algunas observaciones para la optimización.
Ha logrado un desempeño básico, cumpliendo con la tarea de manera
B BÁSICO 2
parcial. Debe corregir algunos aspectos relevantes de la tarea.
No ha logrado cumplir con lo mínimo esperado en el desempeño o no
I INSATISFACTORIO 0
ha completado la tarea.
CATEGORÍA INDICADOR / CRITERIO DE EVALUACIÓN I B S O
Comprensión del Analiza y comprende completamente el caso entregado por el
Caso y Objetivos docente y define claramente el objetivo del modelo.
Análisis Realiza un análisis descriptivo completo y relevante,
Exploratorio de identificando correctamente valores faltantes y outliers.
Datos (EDA) Visualiza y analiza relaciones entre variables para obtener una
comprensión profunda del dataset.
Preprocesamiento Escala y codifica correctamente las variables y maneja los
de Datos datos faltantes de forma adecuada.
Divide el dataset en conjuntos de entrenamiento y prueba,
asegurando una evaluación precisa del modelo.
Selecciona y entrena uno o más algoritmos de Machine
Selección del Learning apropiados para el caso de estudio.
Modelo Optimiza los hiperparámetros del modelo seleccionado para
mejorar su precisión y evitar el sobreajuste.
Evaluación del Compara modelos (si aplica) y selecciona el mejor basado en
Modelo su rendimiento y adecuación al caso de negocio
Interpretación de Analiza la importancia de las variables y explica cómo los
Resultados resultados del modelo impactan la toma de decisiones.
Genera insights claros y aplicables al contexto del caso,
mostrando comprensión en el uso de datos para decisiones.
Documentación y Documenta cada fase del proceso de forma estructurada y
Presentación completa, explicando las decisiones y resultados del modelo.
Realiza visualizaciones efectivas de los resultados, incluyendo
gráficos de desempeño y de importancia de variables.
Implementación y Ofrece recomendaciones prácticas y basadas en datos para
Recomendaciones mejorar el proceso de toma de decisiones en el contexto del
Finales caso.
SUBTOTAL
TOTAL
CALIFICACIÓN
pág. 4

## Página 5

1. RETROALIMENTACIÓN GLOBAL DEL FACILITADOR.
pág. 5