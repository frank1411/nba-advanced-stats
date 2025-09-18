# NBA Advanced Stats - Machine Learning

Este directorio contiene todos los scripts y recursos relacionados con el análisis predictivo y machine learning para las estadísticas avanzadas de la NBA.

## Estructura del directorio

- `models/`: Modelos de machine learning entrenados y serializados.
- `notebooks/`: Jupyter notebooks para análisis exploratorios y desarrollo de modelos.
- `preprocessing/`: Scripts para limpieza, transformación y preparación de datos.
- `utils/`: Funciones auxiliares y módulos reutilizables.

## Cómo empezar

1. Instala las dependencias del proyecto:
   ```bash
   pip install -r ../../requirements.txt
   ```

2. Explora los notebooks en el directorio `notebooks/` para ver ejemplos de análisis.

3. Usa los scripts de preprocesamiento para preparar los datos antes de entrenar los modelos.

## Convenciones

- Nombra tus notebooks con el formato: `[fecha]_[descripción].ipynb` (ej: `2024-09-18_eda_initial.ipynb`)
- Documenta siempre tus modelos con métricas claras y procedimientos de evaluación.
- Mantén los datos sensibles fuera del control de versiones (usar `.gitignore`).
