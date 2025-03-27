import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Folder train fotos
train_path = "../Eigenfaces/train_fotos"


# Cargamos el detector de caras preentrenado de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image, target_size=(200, 200)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No se encontró cara

    # Tomamos la primera cara detectada
    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]

    # Redimensionamos la cara recortada
    face_resized = cv2.resize(face, target_size)
    return face_resized


# Carga los archivos de train
face_images = []
face_filenames = []

for filename in os.listdir(train_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(train_path, filename)
        img = cv2.imread(img_path)
        face = detect_and_crop_face(img)
        if face is not None:
            face_images.append(face)
            face_filenames.append(filename)
# Ver cuantas imagenes se cargaron
#print(len(original_images))



fig, axs = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(cv2.cvtColor(face_images[i], cv2.COLOR_BGR2GRAY), cmap='gray')
    ax.axis('off')
    ax.set_title(f"Cara {i+1}")
plt.tight_layout()
plt.show()


def show_faces_with_names(images, filenames, images_per_row=5):
    total = len(images)
    rows = (total + images_per_row - 1) // images_per_row
    fig, axs = plt.subplots(rows, images_per_row, figsize=(images_per_row * 2.5, rows * 3))

    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].axis('off')
        if i < total:
            axs[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY), cmap='gray')
            axs[i].set_title(filenames[i][:15], fontsize=8)  # Corto por si el nombre es muy largo
    plt.tight_layout()
    plt.show()



show_faces_with_names(face_images, face_filenames)











# Mira cual es el tamaño de las fotos minimo
min_rows, min_cols = sys.maxsize, sys.maxsize
max_rows, max_cols = 0, 0
for (i, image) in enumerate(original_images):
    r, c = image.shape[0], image.shape[1]    
    min_rows = min(min_rows, r)
    max_rows = max(max_rows, r)
    min_cols = min(min_cols, c)
    max_cols = max(max_cols, c)
    
print("\n==> Least common image size:", min_rows, "x", min_cols, "pixels")


# Centrar las imagenes a las fotos
def pad_image_to_size(image, target_rows, target_cols):
    r, c = image.shape[:2]
    top = (target_rows - r) // 2
    bottom = target_rows - r - top
    left = (target_cols - c) // 2
    right = target_cols - c - left
    # Asegurate de que todos los valores sean >= 0
    top, bottom, left, right = max(0, top), max(0, bottom), max(0, left), max(0, right)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)



recentered_images = [pad_image_to_size(img, min_rows, min_cols) for img in original_images]



# Create m x d data matrix
m = len(recentered_images)
d = min_rows * min_cols
X = np.array([img.flatten() for img in recentered_images])


def imshow_gray(img, ax=None):
    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(img, cmap='gray')
        ax.axis('off')


imshow_gray(np.reshape(X[int(len(X)/2), :], (min_rows, min_cols)))

# PCA

U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

# Sanity check on dimensions
print("X:", X.shape)
print("U:", U.shape)
print("Sigma:", Sigma.shape)
print("V^T:", VT.shape)