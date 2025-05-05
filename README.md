# Nuestras Caras

Este repositorio contiene el trabajo práctico "Nuestras Caras", desarrollado para la materia Data Mining Avanzado.
El proyecto se enfoca en la implementación de un sistema de reconocimiento facial, utilizando la técnica de eigenfaces para representar los rostros y una red neuronal multicapa (MLP) que se encarga de su clasificación final.

---

### Autores
- Barbara Noelia Rey
- Maria Guadalupe Salguero
- Noelia Mengoni

---

## Descripción del Proyecto

Este proyecto implementa un sistema de reconocimiento facial utilizando la técnica de Eigenfaces para la representación de rostros, combinada con una red neuronal MLP (Multilayer Perceptron) como clasificador final.

El flujo de trabajo se estructura en tres etapas, organizadas en notebooks:

**1_Procesamiento.ipynb**
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
- Integrar herramientas de procesamiento de imágenes y machine learning en un flujo de trabajo reproducible.

  plaintext
/tarea1-grupo1/
├── /datos_crudos/                             

├── README.md                                  # Documentación del proyecto


## Ejecución del Proyecto

1- Agregar imágenes: colocar las imágenes en formato .jpg o .jpeg dentro de la carpeta:
03_Predecir.ipynb/fotos_nuevas

2- Ejecutar la notebook: abrir y ejecutar el archivo 3_Predecir.ipynb.

3- Verificación de resultados: al finalizar la ejecución, se mostrará el nombre de la persona correspondiente a cada imagen, según la etiqueta predicha por el modelo.
