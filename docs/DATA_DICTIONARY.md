# Diccionario de Datos - Titulados 2007-2024

## Información General

**Fuente**: Ministerio de Educación de Chile  
**Período**: 2007-2024  
**Registros**: 218,566  
**Variables**: 42  
**Archivo**: `TITULADO_2007-2024_web_19_05_2025_E.csv`

---

## Variables por Categoría

### 1. Variables Temporales

| Variable | Tipo | Descripción | Valores | Faltantes |
|----------|------|-------------|---------|-----------|
| `AÑO` | Categórica | Año de titulación | TIT_2007 a TIT_2024 | 0% |

**Notas**: 
- Formato: TIT_YYYY
- Usar para particiones temporales (train/val/test)
- Base para features temporales (tendencias, rezagos)

---

### 2. Variables Geográficas

| Variable | Tipo | Descripción | Valores Únicos | Faltantes |
|----------|------|-------------|----------------|-----------|
| `REGIÓN` | Categórica | Región de la institución | 16 regiones | 0% |
| `PROVINCIA` | Categórica | Provincia | ~56 | <1% |
| `COMUNA` | Categórica | Comuna | ~346 | <1% |

**Notas**:
- Usar para estratificación
- REGIÓN: Variable objetivo potencial
- Útil para análisis de disparidades geográficas

---

### 3. Variables Institucionales

| Variable | Tipo | Descripción | Valores | Faltantes |
|----------|------|-------------|---------|-----------|
| `CÓDIGO INSTITUCIÓN` | Numérica | ID único institución | ~180 | 0% |
| `NOMBRE INSTITUCIÓN` | Categórica | Nombre institución | ~180 | 0% |
| `NOMBRE SEDE` | Categórica | Sede específica | ~600 | <1% |
| `CLASIFICACIÓN INSTITUCIÓN NIVEL 1` | Categórica | Tipo institución | 3 niveles | 0% |
| `CLASIFICACIÓN INSTITUCIÓN NIVEL 2` | Categórica | Subtipo | 5 niveles | 0% |
| `CLASIFICACIÓN INSTITUCIÓN NIVEL 3` | Categórica | Detalle específico | 15 niveles | 0% |

**Valores de Clasificación Nivel 1**:
- Universidades
- Institutos Profesionales
- Centros de Formación Técnica

**Notas**:
- CÓDIGO INSTITUCIÓN: Usar como ID para agrupaciones
- CLASIFICACIÓN: Jerarquía de 3 niveles para análisis multinivel
- NOMBRE SEDE: Útil para análisis por campus

---

### 4. Variables Académicas

| Variable | Tipo | Descripción | Valores | Faltantes |
|----------|------|-------------|---------|-----------|
| `NOMBRE CARRERA` | Categórica | Nombre carrera | ~3,000 | 0% |
| `CÓDIGO PROGRAMA` | Categórica | ID único programa | ~15,000 | 0% |
| `ÁREA DEL CONOCIMIENTO` | Categórica | Área general | 8 áreas | 0% |
| `ÁREA CARRERA GENÉRICA` | Categórica | Área específica | ~50 | <1% |
| `CINE-F_97 ÁREA` | Categórica | Clasificación UNESCO 1997 | 10 | <1% |
| `CINE-F_97 SUBAREA` | Categórica | Subárea UNESCO 1997 | ~30 | <1% |
| `CINE-F_13 ÁREA` | Categórica | Clasificación UNESCO 2013 | 10 | <1% |
| `CINE-F_13 SUBAREA` | Categórica | Subárea UNESCO 2013 | ~30 | <1% |

**Áreas del Conocimiento principales**:
1. Administración y Comercio
2. Agropecuaria
3. Arte y Arquitectura
4. Ciencias Básicas
5. Ciencias Sociales
6. Derecho
7. Educación
8. Humanidades
9. Salud
10. Tecnología

**Notas**:
- ÁREA DEL CONOCIMIENTO: Variable clave para análisis sectorial
- CINE-F: Estándares internacionales (usar 2013 preferentemente)
- Crear features: STEM vs no-STEM

---

### 5. Variables de Modalidad

| Variable | Tipo | Descripción | Valores | Faltantes |
|----------|------|-------------|---------|-----------|
| `NIVEL GLOBAL` | Categórica | Nivel educativo | Pregrado/Postgrado/Postítulo | 0% |
| `CARRERA CLASIFICACIÓN NIVEL 1` | Categórica | Tipo programa | 5 niveles | 0% |
| `CARRERA CLASIFICACIÓN NIVEL 2` | Categórica | Subtipo programa | 10 niveles | 0% |
| `MODALIDAD` | Categórica | Tipo modalidad | Presencial/No Presencial/Semipresencial | 12.2% |
| `JORNADA` | Categórica | Horario | Diurna/Vespertina/A Distancia | <1% |
| `TIPO DE PLAN DE LA CARRERA` | Categórica | Tipo plan | Plan Regular/Continuidad/etc | <1% |

**Distribución NIVEL GLOBAL**:
- Pregrado: 78.8%
- Postítulo: 10.7%
- Posgrado: 10.5%

**Distribución MODALIDAD**:
- Presencial: 81.0%
- Sin información: 12.2%
- No Presencial: 3.9%
- Semipresencial: 3.1%

**Notas**:
- Modalidad tiene 12.2% faltantes → Imputar o crear categoría "Sin Info"
- Útil para analizar tendencias de educación online

---

### 6. Variables de Duración

| Variable | Tipo | Descripción | Rango | Faltantes |
|----------|------|-------------|-------|-----------|
| `DURACIÓN ESTUDIO CARRERA` | Numérica | Duración en semestres | 2-14 | <5% |
| `DURACIÓN TOTAL DE LA CARRERA` | Numérica | Duración total | 2-14 | <5% |

**Notas**:
- Valores en semestres
- Útil para calcular eficiencia (duración real vs nominal)
- Outliers: Revisar valores >12 semestres

---

### 7. Variables de Titulaciones (Target potencial)

| Variable | Tipo | Descripción | Rango | Faltantes |
|----------|------|-------------|-------|-----------|
| `TOTAL TITULACIONES` | Numérica | Total titulados del programa | 1-500+ | 0% |
| `TITULACIONES MUJERES POR PROGRAMA` | Numérica | Tituladas mujeres | 0-400+ | <1% |
| `TITULACIONES HOMBRES POR PROGRAMA` | Numérica | Titulados hombres | 0-400+ | <1% |
| `TITULACIONES NB E INDEFINIDO POR PROGRAMA` | Numérica | No binarios/indefinidos | - | 100% |

**Estadísticas TOTAL TITULACIONES**:
- Media: ~18 titulados/programa
- Mediana: ~10
- Máximo: ~500
- Distribución muy sesgada (cola larga)

**Notas**:
- TITULACIONES NB: 100% faltantes (dato no registrado hasta ahora)
- Crear feature: Ratio mujeres/hombres para análisis de género
- Usar logaritmo para normalizar distribución
- **Variable objetivo principal para modelos predictivos**

---

### 8. Variables de Edad

| Variable | Tipo | Descripción | Rango | Faltantes |
|----------|------|-------------|-------|-----------|
| `TOTAL RANGO EDAD` | Numérica | Total con info edad | 1-500 | 0% |
| `RANGO DE EDAD 15 A 19 AÑOS` | Numérica | Titulados 15-19 | 0-50 | 98.44% |
| `RANGO DE EDAD 20 A 24 AÑOS` | Numérica | Titulados 20-24 | 0-300 | 39.03% |
| `RANGO DE EDAD 25 A 29 AÑOS` | Numérica | Titulados 25-29 | 0-200 | 23.27% |
| `RANGO DE EDAD 30 A 34 AÑOS` | Numérica | Titulados 30-34 | 0-100 | 37.29% |
| `RANGO DE EDAD 35 A 39 AÑOS` | Numérica | Titulados 35-39 | 0-50 | 55.85% |
| `RANGO DE EDAD 40 Y MÁS AÑOS` | Numérica | Titulados 40+ | 0-50 | 58.19% |
| `RANGO DE EDAD SIN INFORMACIÓN` | Numérica | Sin info edad | - | 99.73% |
| `PROMEDIO EDAD PROGRAMA` | Numérica (text) | Edad promedio | 18-60 | <5% |
| `PROMEDIO EDAD MUJER` | Numérica (text) | Edad promedio mujeres | 18-60 | 18.66% |
| `PROMEDIO EDAD HOMBRE` | Numérica (text) | Edad promedio hombres | 18-60 | ~20% |
| `PROMEDIO EDAD NB` | Numérica | Edad promedio NB | - | 100% |

**⚠️ PROBLEMAS DE CALIDAD**:
- Muchas variables de rango etario con >50% faltantes
- PROMEDIO EDAD guardado como texto (comas en lugar de puntos)
- Considerar eliminar rangos con >95% faltantes

**Notas**:
- Convertir promedios de texto a float
- Usar principalmente "RANGO 20-24" y "25-29" (mejor cobertura)
- Crear feature: Edad promedio ponderada
- Útil para identificar programas de educación continua

---

## Variables Derivadas Sugeridas

### Features Temporales
1. **año_numerico**: Convertir TIT_YYYY a número
2. **tendencia_titulaciones**: Cambio año a año
3. **crecimiento_pct**: Variación porcentual
4. **promedio_movil_3años**: Media móvil
5. **es_pandemia**: Flag para 2020-2021

### Features Categóricas
6. **es_STEM**: Booleano para áreas STEM
7. **es_salud**: Booleano para áreas salud
8. **es_universidad**: Tipo institución
9. **es_presencial**: Modalidad presencial
10. **es_postgrado**: Nivel postgrado

### Features de Género
11. **ratio_mujeres**: Mujeres / Total
12. **ratio_hombres**: Hombres / Total
13. **paridad_genero**: Abs(0.5 - ratio_mujeres)
14. **dominio_genero**: Categoría dominante

### Features de Edad
15. **edad_promedio_calc**: Promedio ponderado de rangos
16. **es_adulto_mayor**: >30% de titulados 35+
17. **es_recien_egresado**: >60% de 20-24 años

### Features de Agregación
18. **titulados_region_año**: Total por región/año
19. **titulados_area_año**: Total por área/año
20. **ranking_institucion**: Por total titulados
21. **concentracion_geografica**: Índice Herfindahl por región

### Features de Interacción
22. **region_x_area**: Interacción
23. **institucion_x_modalidad**: Interacción
24. **año_x_nivel**: Interacción temporal

---

## Consideraciones para Modelado

### Variables Objetivo Potenciales
1. **TOTAL TITULACIONES** (regresión)
2. **Clasificación crecimiento** (alto/medio/bajo)
3. **REGIÓN** (clasificación multiclase)
4. **Deserción** (requiere datos adicionales)

### Variables a Eliminar
- TITULACIONES NB: 100% faltantes
- PROMEDIO EDAD NB: 100% faltantes
- RANGO SIN INFORMACIÓN: 99.73% faltantes
- RANGO 15-19 AÑOS: 98.44% faltantes

### Variables a Transformar
- PROMEDIO EDAD PROGRAMA/MUJER/HOMBRE: Texto → Float
- TOTAL TITULACIONES: Aplicar log para normalizar
- AÑO: Extraer número

### Estratificación Recomendada
- Por REGIÓN (mantener balance geográfico)
- Por NIVEL GLOBAL (Pregrado/Postgrado)
- Por AÑO (para validación temporal)

---

## Calidad de Datos - Resumen

| Aspecto | Estado | Acción |
|---------|--------|--------|
| Valores faltantes | Alto en edad/género NB | Imputar o eliminar columnas |
| Tipos de datos | Algunos numéricos como texto | Convertir |
| Outliers | Presentes en titulaciones | Winsorizar o log |
| Desbalance | Regiones/áreas desbalanceadas | Estratificar |
| Encoding | Latin-1 | ✅ Detectado automáticamente |

---

## Referencias

- CINE-F 2013: [UNESCO ISCED Fields](http://uis.unesco.org/en/topic/international-standard-classification-education-isced)
- Diccionario MINEDUC: [Consultar si disponible]

---

**Última actualización**: 2025-10-21
