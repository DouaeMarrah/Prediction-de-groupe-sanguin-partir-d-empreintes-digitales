import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def load_preprocessed_data(data_path):
    """
    Charge les données prétraitées.
    
    Args:
        data_path (str): Chemin vers le fichier de données prétraitées
    
    Returns:
        dict: Données prétraitées
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def scale_features(X_train, X_val, X_test):
    """
    Normalise les caractéristiques.
    
    Args:
        X_train: Données d'entraînement
        X_val: Données de validation
        X_test: Données de test
    
    Returns:
        tuple: Données normalisées
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def optimize_hyperparameters(model, param_grid, X_train, y_train):
    """
    Optimise les hyperparamètres d'un modèle de manière simplifiée.
    
    Args:
        model: Modèle à optimiser
        param_grid: Grille de paramètres
        X_train, y_train: Données d'entraînement
    
    Returns:
        dict: Meilleurs paramètres et score
    """
    # Calcul du nombre total de combinaisons possibles
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    # Utilisation de RandomizedSearchCV avec le nombre approprié d'itérations
    random_search = RandomizedSearchCV(
        model, param_grid, 
        n_iter=min(3, total_combinations),  # Utiliser au maximum 3 itérations ou le nombre total de combinaisons
        cv=3, n_jobs=-1, 
        scoring='accuracy', random_state=42
    )
    random_search.fit(X_train, y_train)
    return {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_
    }

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, classes):
    """
    Entraîne et évalue différents modèles de classification.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        X_test, y_test: Données de test
        classes: Noms des classes
    
    Returns:
        dict: Modèles entraînés et leurs performances
    """
    # Initialisation des modèles et leurs grilles de paramètres simplifiées
    models_config = {
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'params': {
                'n_neighbors': [3, 5],
                'weights': ['uniform']
            }
        }
    }
    
    results = {}
    
    # Entraînement et évaluation de chaque modèle
    for name, config in models_config.items():
        logging.info(f"\nEntraînement du modèle {name}...")
        
        # Optimisation des hyperparamètres
        logging.info("Optimisation des hyperparamètres...")
        optimization_results = optimize_hyperparameters(
            config['model'], config['params'], X_train, y_train
        )
        
        # Configuration du modèle avec les meilleurs paramètres
        model = config['model'].set_params(**optimization_results['best_params'])
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        
        # Entraînement final
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Évaluation
        val_report = classification_report(y_val, y_pred_val, target_names=classes, output_dict=True)
        test_report = classification_report(y_test, y_pred_test, target_names=classes, output_dict=True)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Sauvegarde des résultats
        results[name] = {
            'model': model,
            'val_accuracy': val_report['accuracy'],
            'test_accuracy': test_report['accuracy'],
            'val_report': val_report,
            'test_report': test_report,
            'confusion_matrix': cm,
            'cv_scores': cv_scores.tolist(),
            'best_params': optimization_results['best_params'],
            'optimization_score': optimization_results['best_score']
        }
        
        # Affichage des résultats
        logging.info(f"\nPerformance du modèle {name}:")
        logging.info(f"Meilleurs paramètres: {optimization_results['best_params']}")
        logging.info(f"Score CV moyen: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logging.info(f"Précision sur validation: {val_report['accuracy']:.4f}")
        logging.info(f"Précision sur test: {test_report['accuracy']:.4f}")
    
    return results

def plot_confusion_matrix(cm, classes, model_name, output_dir):
    """
    Trace et sauvegarde la matrice de confusion.
    
    Args:
        cm: Matrice de confusion
        classes: Noms des classes
        model_name: Nom du modèle
        output_dir: Dossier de sortie
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Matrice de confusion - {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    
    # Sauvegarde de la figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def save_results(results, scaler, output_dir):
    """
    Sauvegarde les résultats et le meilleur modèle.
    
    Args:
        results: Résultats des modèles
        scaler: Scaler utilisé
        output_dir: Dossier de sortie
    """
    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des métriques détaillées
    metrics = {}
    for name, result in results.items():
        metrics[name] = {
            'test_accuracy': result['test_accuracy'],
            'val_accuracy': result['val_accuracy'],
            'cv_scores': result['cv_scores'],
            'best_params': result['best_params']
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Sauvegarde du meilleur modèle
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']
    
    model_path = os.path.join(output_dir, f'best_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'model_name': best_model_name,
            'accuracy': results[best_model_name]['test_accuracy'],
            'metrics': metrics[best_model_name]
        }, f)
    
    logging.info(f"\nMeilleur modèle sauvegardé: {best_model_name}")
    logging.info(f"Précision sur l'ensemble de test: {results[best_model_name]['test_accuracy']:.4f}")
    logging.info(f"Modèle sauvegardé dans: {model_path}")
    logging.info(f"Métriques sauvegardées dans: {metrics_path}")

if __name__ == "__main__":
    # Chemins des dossiers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_path = os.path.join(project_dir, 'processed_data', 'preprocessed_dataset.pkl')
    output_dir = os.path.join(project_dir, 'models')
    
    # Chargement des données
    logging.info("Chargement des données prétraitées...")
    data = load_preprocessed_data(data_path)
    
    # Normalisation des caractéristiques
    logging.info("Normalisation des caractéristiques...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        data['X_train'], data['X_val'], data['X_test']
    )
    
    # Entraînement et évaluation des modèles
    logging.info("\nEntraînement des modèles...")
    results = train_and_evaluate_models(
        X_train_scaled, data['y_train'],
        X_val_scaled, data['y_val'],
        X_test_scaled, data['y_test'],
        data['classes']
    )
    
    # Tracé des matrices de confusion
    logging.info("\nGénération des matrices de confusion...")
    for model_name, result in results.items():
        plot_confusion_matrix(
            result['confusion_matrix'],
            data['classes'],
            model_name,
            output_dir
        )
    
    # Sauvegarde des résultats et du meilleur modèle
    save_results(results, scaler, output_dir) 