# ⚠️ IMPORTANTE: REINICIAR KERNEL

El código en `src/data/loader.py` ha sido actualizado, pero Jupyter tiene la versión antigua en memoria.

## Solución: Reiniciar el Kernel

### Opción 1: Desde el menú (RECOMENDADO)
1. Click en **Kernel** → **Restart Kernel...** 
2. Confirmar
3. Ejecutar todas las celdas de nuevo desde el principio

### Opción 2: Atajo de teclado
- Presiona `00` (cero dos veces rápido)
- Ejecutar todas las celdas de nuevo

### Opción 3: Agregar esta celda al inicio del notebook

```python
# Forzar recarga de módulos modificados
%load_ext autoreload
%autoreload 2
```

Luego ejecutar normalmente.

---

## ¿Por qué pasa esto?

Python guarda los módulos importados en caché. Cuando modificas un archivo `.py`, 
Jupyter no lo recarga automáticamente hasta que reinicies el kernel.
