import polars as pl
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from IPython import display
import time
import os
import pickle
from functools import reduce
import cv2
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
import shutil
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelBinarizer

"""## **Cargar modelo DNN para la detección de caras**"""

# Cargar el modelo DNN para detección de caras
MODELO_DIR = "/content/drive/MyDrive/DMA_Eigenfaces/"
PROTOTXT_PATH = os.path.join(MODELO_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELO_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def verificar_y_descargar_modelos():
    os.makedirs(MODELO_DIR, exist_ok=True)
    if not os.path.exists(PROTOTXT_PATH):
        os.system(f'wget -q -O {PROTOTXT_PATH} https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt')
    if not os.path.exists(CAFFEMODEL_PATH):
        os.system(f'wget -q -O {CAFFEMODEL_PATH} https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel')


def cargar_face_detector() -> cv2.dnn_Net:
    verificar_y_descargar_modelos()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    return net

"""## **Función para detectar los rostros**"""

def detectar_cara_dnn(img: np.ndarray, net: cv2.dnn_Net, confidence_threshold: float) -> Optional[Tuple[int, int, int, int]]:
    """
    Detecta una cara y devuelve las coordenadas de la mejor detección como (x1, y1, x2, y2).
    No realiza recortes, ni cambios de color, ni resize.
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if detections.shape[2] == 0:
        return None

    best_conf, best_box = 0, None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf and conf > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            best_conf = conf

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return (x1, y1, x2, y2)

"""## **Función para procesar las imagenes y guardarlas redimensionadas en disco.**"""

def procesar_y_crear_dataset(
    input_folder: str,
    output_folder: str,
    sin_caras_folder: Optional[str],
    pickle_path: Optional[str],
    net: cv2.dnn_Net,
    img_size: Tuple[int, int] = (64, 64),
    confidence_threshold: float = 0.3
) -> pl.DataFrame:
    """
    Procesa imágenes: detecta caras, recorta, convierte a gris, resizea, guarda en disco y crea un DataFrame Polars.
    Opcionalmente, guarda el DataFrame en un archivo Pickle.

    Args:
        input_folder: Carpeta con subcarpetas de imágenes etiquetadas.
        output_folder: Carpeta donde se guardarán las imágenes procesadas.
        sin_caras_folder: Carpeta para imágenes sin caras detectadas (opcional).
        pickle_path: Ruta para guardar el DataFrame como archivo Pickle (opcional).
        net: Modelo DNN para detección de caras.
        img_size: Tamaño al que se redimensionan las imágenes.
        confidence_threshold: Umbral de confianza para detección de caras.

    Returns:
        Polars DataFrame con dos columnas: 'imagen' (array aplanado) y 'etiqueta' (nombre de la persona).
    """
    imagenes = []
    etiquetas = []

    for subdir, _, files in os.walk(input_folder):
        etiqueta = os.path.basename(subdir)
        if subdir == input_folder:
            continue

        output_subdir = os.path.join(output_folder, etiqueta)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_path = os.path.join(subdir, file)
            img = cv2.imread(input_path)

            if img is None:
                continue

            box = detectar_cara_dnn(img, net, confidence_threshold)
            if box is not None:
                x1, y1, x2, y2 = box

                # Añadir margen
                margin_ratio = 0.2
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(bw * margin_ratio))
                y1 = max(0, y1 - int(bh * margin_ratio))
                x2 = min(img.shape[1], x2 + int(bw * margin_ratio))
                y2 = min(img.shape[0], y2 + int(bh * margin_ratio))

                face_crop = img[y1:y2, x1:x2]
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = cv2.resize(face_crop, img_size)

                # Guardar imagen procesada
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, face_crop)

                # Aplanar imagen para el DataFrame
                img_flat = face_crop.flatten()
                imagenes.append(img_flat)
                etiquetas.append(etiqueta)

            else:
                if sin_caras_folder:
                    os.makedirs(sin_caras_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(sin_caras_folder, file), img)

    df = pl.DataFrame({
        "imagen": imagenes,
        "etiqueta": etiquetas
    })

    # Guardar DataFrame como Pickle si se especifica pickle_path
    if pickle_path:
        os.makedirs(os.path.dirname(pickle_path) or '.', exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)

    return df