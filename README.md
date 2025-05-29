```🛒 Predicción de la Demanda en Retail con Machine Learning```

¿Por qué predecir la demanda?

Los supermercados deben anticipar cuántos productos se venderán en el futuro para evitar desabastecimientos o sobrestock. Este proyecto desarrolla un modelo de predicción de ventas por tienda y producto a partir de datos históricos, con el objetivo de apoyar la toma de decisiones operativas.

📌 Objetivo

Construir una solución basada en Machine Learning que permita predecir la demanda futura en el sector retail, combinando modelos avanzados y una interfaz visual interactiva.

🔍 Fase 1 – Prueba de Concepto (PoC)
```
· Definición del problema
· Exploración de datos sintéticos
· Preprocesamiento básico
· Ingeniería de características
· Entrenamiento con Random Forest
· Evaluación inicial del rendimiento
· Análisis de resultados y validación del enfoque
```

🎯 Objetivo: Validar la viabilidad técnica con datos simulados.

🧠 Fase 2 – Desarrollo con Datos Reales
```
· Integración de datos reales de Walmart
· Limpieza y estructuración de datos
· Feature engineering (lags, medias móviles, etc.)
· Comparación de modelos: Random Forest, XGBoost, otros
· Selección final del modelo: XGBoost
· Evaluación con métricas: MAE, MSE, R²
```
🎯 Objetivo: Implementar el modelo óptimo con datos reales.

💻 Fase 3 – Implementación Final con Streamlit
```
· Desarrollo de una app interactiva con Streamlit
· Visualización de predicciones por tienda y producto
· Filtros y selección de escenarios personalizados
· Presentación del modelo como herramienta para usuarios no técnicos
```
🎯 Objetivo: Entregar una herramienta funcional para el usuario final.

🛠️ Tecnologías utilizadas
```
· Python (Pandas, NumPy, Scikit-learn, XGBoost, Regresión Lineal, Random Forest, Prophet)
· Visualización: Matplotlib, Seaborn
· Desarrollo web: Streamlit
· Evaluación de modelos: MSE, MAE, R²
· Web Scraping
```

📂 Estructura del repositorio
```
MACHINE_LEARNING_BASED_RETAIL_DEMAND/
│
├── POC/                     # Fase 1: Prueba de concepto con datos sintéticos
│   ├── data/
│   ├── models/
│   └── src/
│
├── Walmart/                 # Fase 2: Desarrollo con datos reales de Walmart
│   ├── data/                # Datos estructurados, procesados y crudos
│   │   ├── csv_model/
│   │   ├── data_base/
│   │   ├── excels/
│   │   ├── jsons/
│   │   ├── norm_scal/
│   │   └── raw/
│   │       ├── extractions/
│   │       └── macrodata/
│   │
│   ├── models/              # Modelos entrenados finales
│   │
│   └── src/                 # Scripts y notebooks de desarrollo
│       ├── Data_extraction/
│       └── Web Scraping/
│
├── .gitignore               # Archivos a ignorar por Git
├── LICENSE                  # Licencia del repositorio
└── README.md                # Documentación principal
```