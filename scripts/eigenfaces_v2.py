import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def detectar_y_recortar_cara(image_path, output_path, img_size=(30, 30)):
    # Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen: {image_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cargar el clasificador Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Ajuste de parámetros para mejorar detección
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(faces) == 0:
        print(f"No se detectó ninguna cara en {image_path}")
        return False
    
    # Seleccionar la cara más grande (por si hay más de una)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Validar tamaño mínimo de la cara detectada
    if w < 30 or h < 30:
        print(f"Cara demasiado pequeña en {image_path}")
        return False

    # Recortar y redimensionar la cara
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, img_size)

    # Guardar la imagen procesada
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    cv2.imwrite(output_path, face_resized)
    print(f"Cara guardada en: {output_path}")
    return True

# Procesar imágenes en una carpeta
def procesar_carpeta(input_folder, output_folder, img_size=(30, 30)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(subdir, file)
            output_subdir = os.path.join(output_folder, os.path.basename(subdir))

            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            output_path = os.path.join(output_subdir, file)
            detectar_y_recortar_cara(input_path, output_path, img_size)

# Ejecutar
input_folder = '../Eigenfaces/Eigenfaces'
output_folder = '../Eigenfaces/procesadas2'
procesar_carpeta(input_folder, output_folder)

def cargar_imagenes(input_folder, img_size=(30, 30)):
    X = []
    y = []
    label_dict = {}
    label_counter = 0

    # Recorrer las carpetas de imágenes procesadas
    for subdir, _, files in os.walk(input_folder):
        if not files:
            continue
        if subdir not in label_dict:
            label_dict[subdir] = label_counter
            label_counter += 1

        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                X.append(img_resized.flatten())
                y.append(label_dict[subdir])

    print(f"Se cargaron {len(X)} imágenes.")
    return np.array(X), np.array(y)

# Cargar imágenes procesadas
input_folder = '../Eigenfaces/procesadas'
X, y = cargar_imagenes(input_folder)

# Estandarizar las imágenes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensiones originales: {X.shape[1]}")
print(f"Dimensiones después de PCA: {X_pca.shape[1]}")

# Visualizar la varianza acumulada
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.title('Varianza Explicada por PCA')
plt.show()

mean_face = np.mean(X, axis=0)

# Convertir a imagen 2D
mean_face_image = mean_face.reshape(30, 30)

# Visualizar la cara promedio
plt.figure(figsize=(6, 6))
plt.imshow(mean_face_image, cmap='gray')
plt.title("Cara Promedio")
plt.axis('off')
plt.show()

