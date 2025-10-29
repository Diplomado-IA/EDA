# Guía de Desarrollo del Proyecto EDA

## 1. Principios Fundamentales

### 1.1 Cultura de Equipo
- **Colaboración Humano-IA**: La IA es un miembro activo del equipo de desarrollo
- **Comunicación Clara**: Documentar decisiones y cambios de manera concisa
- **Calidad sobre Velocidad**: Priorizar código limpio y mantenible
- **Aprendizaje Continuo**: Compartir conocimientos entre todos los miembros

### 1.2 Responsabilidad Compartida
- Todos los miembros (humanos e IA) son responsables de la calidad del código
- Revisión de código obligatoria antes de commits importantes
- La IA debe explicar sus decisiones cuando se le solicite

## 2. Estándares de Código

### 2.1 Python
- **PEP 8**: Seguir las convenciones de estilo de Python
- **Nomenclatura**:
  - Variables y funciones: `snake_case`
  - Clases: `PascalCase`
  - Constantes: `UPPER_CASE`
- **Documentación**: Docstrings para todas las funciones y clases
- **Type Hints**: Usar anotaciones de tipo cuando sea posible

```python
def procesar_datos(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Procesa los datos de un DataFrame específico.
    
    Args:
        df: DataFrame a procesar
        columna: Nombre de la columna objetivo
    
    Returns:
        DataFrame procesado
    """
    pass
```

### 2.2 Estructura de Archivos
```
EDA/
├── data/           # Datos crudos (no versionados si son grandes)
├── notebooks/      # Jupyter notebooks para exploración
├── src/           # Código fuente reutilizable
│   ├── utils/     # Funciones utilitarias
│   ├── models/    # Modelos y clases
│   └── analysis/  # Análisis específicos
├── docs/          # Documentación del proyecto
└── tests/         # Tests unitarios
```

## 3. Control de Versiones (Git)

### 3.1 Commits
- **Mensajes descriptivos**: Usar formato convencional
  ```
  tipo(alcance): descripción breve
  
  feat: Nueva funcionalidad
  fix: Corrección de bug
  docs: Cambios en documentación
  refactor: Refactorización de código
  test: Añadir o modificar tests
  chore: Tareas de mantenimiento
  ```

- **Commits atómicos**: Un commit = un cambio lógico
- **Frecuencia**: Commits pequeños y frecuentes

### 3.2 Branches
- `main`: Código estable y funcional
- `develop`: Desarrollo activo
- `feature/<nombre>`: Nuevas funcionalidades
- `fix/<nombre>`: Correcciones

### 3.3 Prohibido
- Commits de archivos grandes de datos
- Commits de credenciales o API keys
- Commits directos a `main` sin revisión

## 4. Gestión de Datos

### 4.1 Datos Sensibles
- Nunca versionar datos personales o sensibles
- Usar `.gitignore` para excluir archivos de datos grandes
- Documentar fuentes de datos en `README.md`

### 4.2 Formato y Almacenamiento
- Datos crudos en `data/raw/`
- Datos procesados en `data/processed/`
- Usar formatos eficientes (parquet, feather) para datos grandes
- Mantener datos originales inmutables

## 5. Análisis Exploratorio de Datos

### 5.1 Notebooks
- **Estructura clara**: 
  1. Importaciones
  2. Carga de datos
  3. Exploración inicial
  4. Análisis detallado
  5. Conclusiones

- **Nombrar notebooks**: `01_exploracion_inicial.ipynb`, `02_limpieza_datos.ipynb`
- **Limpiar outputs**: Antes de commits (usar `nbstripout` o similar)

### 5.2 Visualizaciones
- Títulos y etiquetas claros en todas las gráficas
- Paletas de colores accesibles
- Guardar figuras importantes en `docs/figures/`

## 6. Colaboración con IA

### 6.1 Uso Efectivo de IA
- **Contexto claro**: Proporcionar información suficiente para la tarea
- **Iteración**: Refinar solicitudes basándose en resultados
- **Validación**: Siempre revisar código generado por IA
- **Documentación**: La IA debe documentar su código

### 6.2 Responsabilidades de la IA
- Generar código limpio y bien documentado
- Sugerir mejores prácticas cuando sea apropiado
- Explicar decisiones técnicas cuando se solicite
- Minimizar cambios - solo modificar lo necesario
- Validar que los cambios no rompan funcionalidad existente

### 6.3 Responsabilidades Humanas
- Definir requisitos claros
- Revisar y validar código generado
- Tomar decisiones de arquitectura
- Proporcionar feedback constructivo

## 7. Calidad y Testing

### 7.1 Tests
- Tests unitarios para funciones críticas
- Validación de transformaciones de datos
- Tests de integración para pipelines completos

### 7.2 Revisión de Código
- **Checklist**:
  - ¿El código es legible?
  - ¿Está documentado?
  - ¿Sigue las convenciones?
  - ¿Maneja errores apropiadamente?
  - ¿Es eficiente?
  - ¿Cumple con estandar de seguridad OWASP Top Ten 2025?

## 8. Documentación

### 8.1 README.md
- Propósito del proyecto
- Instalación y configuración
- Estructura del proyecto
- Uso básico
- Contribuidores

### 8.2 Documentación de Código
- Docstrings en funciones complejas
- Comentarios para lógica no obvia
- No comentar lo obvio

### 8.3 Documentación de Decisiones
- Mantener registro de decisiones importantes
- Documentar cambios de arquitectura
- Justificar elecciones técnicas

## 9. Dependencias

### 9.1 Gestión de Paquetes
- `requirements.txt`: Dependencias de producción
- Especificar versiones: `pandas==2.0.0`
- Mantener actualizado

### 9.2 Entorno Virtual
- Siempre usar entorno virtual
- Documentar versión de Python
- Proveer instrucciones de instalación

## 10. Seguridad y Privacidad

### 10.1 Datos
- No compartir datos sensibles con sistemas externos
- Anonimizar datos cuando sea posible
- Cumplir con regulaciones de privacidad

### 10.2 Código
- No incluir credenciales en código
- Usar variables de entorno para configuración
- Revisar dependencias por vulnerabilidades

## 11. Comunicación

### 11.1 Issues y Tareas
- Usar sistema de tracking (GitHub Issues, TODO.md)
- Descripción clara del problema o tarea
- Asignar prioridades

### 11.2 Retrospectivas
- Revisar progreso regularmente
- Identificar mejoras en el proceso
- Celebrar logros del equipo

## 12. Proceso de Desarrollo

### 12.1 Workflow Estándar
1. Crear issue/tarea
2. Crear branch desde `develop`
3. Desarrollar y hacer commits
4. Ejecutar tests
5. Crear Pull Request
6. Revisión de código
7. Merge a `develop`
8. Deploy a `main` cuando sea estable

### 12.2 Definición de "Hecho"
- Código implementado y funcional
- Tests pasando
- Documentación actualizada
- Revisión completada
- Sin deuda técnica conocida

## 13. Mejores Prácticas Específicas de EDA

### 13.1 Exploración
- Entender el dominio del problema primero
- Validar calidad de datos desde el inicio
- Documentar insights y hallazgos

### 13.2 Reproducibilidad
- Fijar seeds aleatorias
- Documentar versiones de librerías
- Scripts que se puedan ejecutar de inicio a fin

### 13.3 Eficiencia
- Trabajar con muestras para exploración inicial
- Optimizar código antes de escalar
- Monitorear uso de memoria

---

## Actualizaciones

Este documento es vivo y debe actualizarse conforme el equipo evoluciona.

**Última actualización**: 2025-10-21

**Responsable**: Equipo de Desarrollo (Humanos + IA)
