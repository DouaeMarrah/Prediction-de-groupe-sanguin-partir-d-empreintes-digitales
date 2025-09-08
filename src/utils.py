import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage.morphology import skeletonize
from sklearn.preprocessing import LabelEncoder

def load_image(image_path, target_size=(256, 256)):
    """
    Charge et redimensionne une image.
    
    Args:
        image_path (str): Chemin vers l'image
        target_size (tuple): Taille cible de l'image
    
    Returns:
        numpy.ndarray: Image redimensionnée
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    return cv2.resize(img, target_size)

def preprocess_image(image):
    """
    Prétraite une image d'empreinte digitale.
    
    Args:
        image (numpy.ndarray): Image d'entrée
    
    Returns:
        numpy.ndarray: Image prétraitée
    """
    # Normalisation
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_norm)
    
    # Réduction du bruit
    img_denoised = cv2.fastNlMeansDenoising(img_clahe)
    
    # Binarisation
    _, img_binary = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_binary

def extract_lbp_features(image, P=8, R=1):
    """
    Extrait les caractéristiques LBP d'une image.
    
    Args:
        image (numpy.ndarray): Image d'entrée
        P (int): Nombre de points
        R (int): Rayon
    
    Returns:
        numpy.ndarray: Caractéristiques LBP
    """
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return hist

def extract_hog_features(image):
    """
    Extrait les caractéristiques HOG d'une image.
    
    Args:
        image (numpy.ndarray): Image d'entrée
    
    Returns:
        numpy.ndarray: Caractéristiques HOG
    """
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False)
    return features

def extract_minutiae_features(image):
    """
    Extrait les caractéristiques des minuties d'une image.
    
    Args:
        image (numpy.ndarray): Image d'entrée
    
    Returns:
        numpy.ndarray: Caractéristiques des minuties
    """
    # Squelettisation
    skeleton = skeletonize(image > 0)
    
    # Détection des minuties
    minutiae = []
    for i in range(1, skeleton.shape[0]-1):
        for j in range(1, skeleton.shape[1]-1):
            if skeleton[i,j]:
                # Extraction du voisinage 3x3
                neighborhood = skeleton[i-1:i+2, j-1:j+2].astype(np.uint8)
                # Calcul du nombre de transitions
                transitions = np.sum(np.abs(np.diff(neighborhood.flatten())))
                
                if transitions == 2:  # Terminaison
                    minutiae.append([i, j, 1])
                elif transitions == 6:  # Bifurcation
                    minutiae.append([i, j, 2])
    
    return np.array(minutiae) if minutiae else np.zeros((1, 3))

def extract_all_features(image):
    """
    Extrait toutes les caractéristiques d'une image.
    
    Args:
        image (numpy.ndarray): Image d'entrée
    
    Returns:
        numpy.ndarray: Vecteur de caractéristiques combinées
    """
    # Prétraitement
    preprocessed_img = preprocess_image(image)
    
    # Extraction des caractéristiques
    lbp_features = extract_lbp_features(preprocessed_img)
    hog_features = extract_hog_features(preprocessed_img)
    minutiae_features = extract_minutiae_features(preprocessed_img)
    
    # Aplatissement des caractéristiques des minuties
    minutiae_flat = minutiae_features.flatten()
    
    # Combinaison des caractéristiques
    return np.concatenate([
        lbp_features,
        hog_features,
        minutiae_flat
    ])

def load_dataset(data_dir):
    """
    Charge le dataset complet.
    
    Args:
        data_dir (str): Chemin vers le dossier de données
    
    Returns:
        tuple: (X, y) données et labels
    """
    X = []
    y = []
    
    for blood_group in os.listdir(data_dir):
        blood_group_path = os.path.join(data_dir, blood_group)
        if os.path.isdir(blood_group_path):
            for img_file in os.listdir(blood_group_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(blood_group_path, img_file)
                        img = load_image(img_path)
                        features = extract_all_features(img)
                        X.append(features)
                        y.append(blood_group)
                    except Exception as e:
                        print(f"Erreur lors du traitement de {img_file}: {str(e)}")
    
    return np.array(X), np.array(y) 