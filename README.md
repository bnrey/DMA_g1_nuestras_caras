# Nuestras Caras

Este repositorio contiene el trabajo práctico "Nuestras Caras", realizado en el marco de la asignatura Data Mining Avanzado. El proyecto tiene como eje la implementación de un sistema de reconocimiento facial, empleando la técnica de Isomap para la reducción de dimensionalidad y selección de características, junto con el algoritmo de Backpropagation para entrenar una red neuronal multicapa. El objetivo fue detectar y clasificar los rostros de los estudiantes de la materia.

---

### Autores
- Barbara Noelia Rey
- Maria Guadalupe Salguero
- Noelia Mengoni

---

## Descripción del Proyecto

El trabajo se estructura en tres etapas, organizadas en notebooks:

**1_Procesar.ipynb**
Se realiza el procesamiento inicial de las imágenes. Se detectan los rostros utilizando un modelo preentrenado de OpenCV (DNN), se recortan y estandarizan. Luego, se aplica el algoritmo Isomap para reducir la dimensionalidad y generar un conjunto de variables numéricas representativas de cada rostro.

**2_Entrenar.ipynb**
Se entrena una red neuronal MLP con los datos procesados. Se prueban distintas configuraciones: número de capas ocultas, cantidad de neuronas, funciones de activación y semillas aleatorias. Se calculan métricas de error en el conjunto de entrenamiento y se guarda el modelo final para su posterior aplicación.

**3_Predecir.ipynb**
Se cargan los modelos entrenados para predecir sobre nuevas imágenes. Se evalúa el rendimiento sobre datos de test, se calcula la tasa de error y se genera un informe con los resultados de clasificación. Además, se visualiza el desempeño comparando clases reales y predichas.


## Objetivos

- Desarrollar un sistema capaz de reconocer rostros a partir de imágenes digitales.
- Aplicar técnicas de reducción de dimensionalidad (Isomap) para representar las imágenes en un espacio compacto.
- Entrenar una red neuronal MLP para clasificar los rostros con alta precisión.
- Evaluar el rendimiento del modelo sobre datos de test.

## Estructura del Proyecto

La estructura de archivos en el repositorio es la siguiente:

```plaintext
Github/
└── DMA_Eigenfaces/
    ├── notebooks/
    │   ├── 1_Procesar.ipynb
    │   ├── 2_Entrenar.ipynb
    │   └── 3_Predecir.ipynb
    ├── src/
    │   ├── procesar_imagenes.py
    │   └── multiperceptron.py
    ├── modelos/
    │       ├── data.pkl
    │       └── red.pkl
    │       ├── isomap.pkl
    │       └── scaler.pkl
    └── README.md
├── README.md  
```

## Ejecución del Proyecto

1. Carga y procesamiento de nuevas imágenes
Las imágenes se cargan manualmente desde el entorno de ejecución. A continuación, se detectan los rostros y se aplican los mismos pasos de preprocesamiento utilizados en el dataset original, incluyendo la estandarización y la proyección al espacio reducido mediante Isomap.

2. Ejecución de la notebook de predicción
Se debe abrir y ejecutar el archivo 3_Predecir.ipynb, el cual utiliza el modelo de red neuronal multiperceptrón previamente entrenado y guardado en disco para realizar las predicciones sobre las nuevas imágenes.

3. Visualización de resultados
Como resultado, se genera un nuevo dataframe que contiene la predicción para cada imagen procesada, indicando a qué persona fue asignada o, en su defecto, si fue clasificada como intruso.
