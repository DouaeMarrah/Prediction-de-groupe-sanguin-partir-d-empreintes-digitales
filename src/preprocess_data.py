import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from skimage.feature import local_binary_pattern, hog

def extract_features(image):
    """
    Extrait différentes caractéristiques d'une image.
    
    Args:
        image: Image en niveaux de gris
    
    Returns:
        np.array: Vecteur de caractéristiques
    """
    features = []
    
    # 1. Caractéristiques HOG (Histogram of Oriented Gradients)
    # Réduire la complexité du HOG
    hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=False)
    features.extend(hog_features)
    
    # 2. Local Binary Patterns
    # Réduire la complexité du LBP
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                              range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)
    features.extend(hist_lbp)
    
    # 3. Statistiques basiques de l'image
    features.extend([
        np.mean(image),      # Moyenne
        np.std(image),       # Écart-type
        np.median(image),    # Médiane
        np.max(image),       # Maximum
        np.min(image)        # Minimum
    ])
    
    return np.array(features)

def load_and_preprocess_image(image_path, target_size=(64, 64)):  # Réduire la taille de l'image
    """
    Charge et prétraite une image.
    
    Args:
        image_path (str): Chemin vers l'image
        target_size (tuple): Taille cible pour le redimensionnement
    
    Returns:
        np.array: Image prétraitée
    """
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Redimensionner l'image
    img_resized = cv2.resize(img, target_size)
    
    # Normaliser les valeurs des pixels entre 0 et 1
    img_normalized = img_resized.astype('float32') / 255.0
    
    return img_normalized

def prepare_dataset(data_dir, target_size=(128, 128), test_size=0.2, val_size=0.2):
    """
    Prépare le dataset pour l'entraînement.
    
    Args:
        data_dir (str): Chemin vers le dossier de données
        target_size (tuple): Taille cible pour les images
        test_size (float): Proportion des données pour le test
        val_size (float): Proportion des données pour la validation
    
    Returns:
        dict: Dictionnaire contenant les caractéristiques et labels divisés
    """
    # Groupes sanguins possibles
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    # Listes pour stocker les données
    features = []
    labels = []
    
    print("Chargement et extraction des caractéristiques des images...")
    total_images = 0
    
    # Parcourir chaque groupe sanguin
    for label, group in enumerate(blood_groups):
        group_dir = os.path.join(data_dir, group)
        if not os.path.exists(group_dir):
            continue
            
        # Parcourir les images du groupe
        group_files = [f for f in os.listdir(group_dir) if f.lower().endswith('.bmp')]
        print(f"\nTraitement du groupe {group}: {len(group_files)} images")
        
        for i, img_file in enumerate(group_files, 1):
            if i % 100 == 0:
                print(f"Progression {group}: {i}/{len(group_files)} images")
                
            img_path = os.path.join(group_dir, img_file)
            try:
                # Prétraiter l'image
                img_processed = load_and_preprocess_image(img_path, target_size)
                # Extraire les caractéristiques
                img_features = extract_features(img_processed)
                features.append(img_features)
                labels.append(label)
                total_images += 1
            except Exception as e:
                print(f"Erreur lors du traitement de {img_file}: {str(e)}")
    
    # Convertir en arrays numpy
    X = np.array(features)
    y = np.array(labels)
    
    # Première division : séparer les données de test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Deuxième division : séparer les données d'entraînement et de validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    # Créer le dictionnaire des données
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'classes': blood_groups,
        'feature_names': [
            'hog_features',
            'lbp_histogram',
            'mean',
            'std',
            'median',
            'max',
            'min'
        ]
    }
    
    print(f"\nNombre total d'images traitées: {total_images}")
    print("\nStatistiques du dataset prétraité:")
    print(f"Nombre de caractéristiques par image: {X.shape[1]}")
    print(f"Données d'entraînement: {X_train.shape[0]} images")
    print(f"Données de validation: {X_val.shape[0]} images")
    print(f"Données de test: {X_test.shape[0]} images")
    
    return dataset

def save_dataset(dataset, output_dir):
    """
    Sauvegarde le dataset prétraité.
    
    Args:
        dataset (dict): Dataset prétraité
        output_dir (str): Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'preprocessed_dataset.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset prétraité sauvegardé dans: {output_path}")

if __name__ == "__main__":
    # Chemins des dossiers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(project_dir, 'processed_data')
    
    # Préparer et sauvegarder le dataset
    dataset = prepare_dataset(data_dir)
    save_dataset(dataset, output_dir) 