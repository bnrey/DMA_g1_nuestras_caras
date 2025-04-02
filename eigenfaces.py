import sys
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Ruta donde están las imágenes
train_image_folder = "..Eigenfaces/train_fotos"
test_image_folder = "..Eigenfaces/test_fotos"

# Dimensión fija para todas las imágenes
IMG_SIZE = (64, 64)

# Listas para almacenar datos
X = []  # Características (imágenes aplanadas)
y = []  # Etiquetas (nombre de la clase de cada imagen)

# Leer imágenes desde la carpeta
for label, folder in enumerate(os.listdir(train_image_folder)):  # Suponiendo que cada clase tiene su propia carpeta
    folder_path = os.path.join(train_image_folder, folder)
    
    if os.path.isdir(folder_path):  # Verificar que es una carpeta
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            
            # Cargar la imagen en escala de grises y redimensionar
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Omitir archivos no válidos
            
            img = cv2.resize(img, IMG_SIZE)  # Redimensionar a 64x64
            
            X.append(img.flatten())  # Convertir la imagen en un vector 1D
            y.append(label)  # Usar el índice de la carpeta como etiqueta de la clase

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar PCA
n_components = 50  # Número de componentes principales
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entrenar un clasificador k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

# Evaluación del modelo
accuracy = knn.score(X_test_pca, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")



min_rows, min_cols = sys.maxsize, sys.maxsize
max_rows, max_cols = 0, 0
for (i, image) in enumerate(original_images):
    r, c = image.shape[0], image.shape[1]    
    min_rows = min(min_rows, r)
    max_rows = max(max_rows, r)
    min_cols = min(min_cols, c)
    max_cols = max(max_cols, c)
    
print("\n==> Least common image size:", min_rows, "x", min_cols, "pixels")