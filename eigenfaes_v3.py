import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def cargar_imagenes(directorio):
    """
    Carga imágenes de un directorio, conviertiéndolas a escala de grises y redimensionándolas.
    
    Parámetros:
    - directorio: Ruta al directorio con las imágenes de entrenamiento
    
    Retorna:
    - imagenes: Lista de imágenes procesadas
    - etiquetas: Lista de etiquetas correspondientes
    - nombres: Lista de nombres de personas
    """
    imagenes = []
    etiquetas = []
    nombres = []
    
    # Iterar sobre subdirectorios (cada subdirectorio representa una persona)
    for nombre in os.listdir(directorio):
        ruta_persona = os.path.join(directorio, nombre)
        
        # Verificar que sea un directorio
        if os.path.isdir(ruta_persona):
            for archivo in os.listdir(ruta_persona):
                ruta_imagen = os.path.join(ruta_persona, archivo)
                
                # Cargar imagen
                img = cv2.imread(ruta_imagen)
                if img is not None:
                    # Convertir a escala de grises
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Redimensionar
                    gray_resized = cv2.resize(gray, (64, 64))
                    
                    imagenes.append(gray_resized)
                    etiquetas.append(len(nombres))
            
            nombres.append(nombre)
    
    return np.array(imagenes), np.array(etiquetas), nombres

def entrenar_modelo_facial(directorio_entrenamiento):
    """
    Entrena un modelo de reconocimiento facial usando Eigenfaces y KNN.
    
    Parámetros:
    - directorio_entrenamiento: Ruta al directorio con imágenes de entrenamiento
    
    Retorna:
    - scaler: Objeto StandardScaler
    - pca: Objeto PCA
    - knn: Clasificador KNN entrenado
    - label_dict: Diccionario de mapeo de etiquetas
    """
    # Cargar imágenes
    X, y, nombres = cargar_imagenes(directorio_entrenamiento)
    
    # Aplanar imágenes
    X_flattened = X.reshape(X.shape[0], -1)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Aplicar PCA
    pca = PCA(n_components=0.95)  # Mantener 95% de la varianza
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Entrenar KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    
    # Crear diccionario de etiquetas
    label_dict = {nombre: i for i, nombre in enumerate(nombres)}
    
    return scaler, pca, knn, label_dict

def predecir_cara(nueva_imagen_path, scaler, pca, knn, label_dict):
    """
    Predice la identidad de una cara en una imagen.
    
    Parámetros:
    - nueva_imagen_path: Ruta a la imagen de prueba
    - scaler: Objeto StandardScaler entrenado
    - pca: Objeto PCA entrenado
    - knn: Clasificador KNN entrenado
    - label_dict: Diccionario de mapeo de etiquetas
    
    Retorna:
    - Nombre de la persona predicha y probabilidad de confianza
    """
    # Cargar imagen
    img = cv2.imread(nueva_imagen_path)
    if img is None:
        print("Error al cargar la imagen.")
        return None, None
    
    # Convertir a escala de grises y redimensionar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (64, 64))
    
    # Aplanar la imagen
    img_flatten = gray_resized.flatten().reshape(1, -1)
    
    # Estandarizar
    img_scaled = scaler.transform(img_flatten)
    
    # Aplicar PCA
    img_pca = pca.transform(img_scaled)
    
    # Hacer predicción
    prediccion = knn.predict(img_pca)
    proba = knn.predict_proba(img_pca).max()
    
    # Obtener nombre de la clase
    nombre_clase = [nombre for nombre, clase in label_dict.items() if clase == prediccion[0]][0]
    
    # Mostrar imagen
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_resized, cmap='gray')
    plt.title(f"Predicción: {nombre_clase} (Confianza: {proba:.2f})")
    plt.axis('off')
    plt.show()
    
    print(f"La imagen fue identificada como: {nombre_clase}")
    return nombre_clase, proba

# Ejemplo de uso
if __name__ == "__main__":
    # Directorio con imágenes de entrenamiento
    directorio_entrenamiento = "../Eigenfaces/Eigenfaces"
    
    # Entrenar modelo
    scaler, pca, knn, label_dict = entrenar_modelo_facial(directorio_entrenamiento)
    
    # Ruta a imagen de prueba
    nueva_imagen_path = "../Eigenfaces/test/test6.jpg"
    # Predecir cara
    predecir_cara(nueva_imagen_path, scaler, pca, knn, label_dict)