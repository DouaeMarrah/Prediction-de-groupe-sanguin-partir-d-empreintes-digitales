import os
import shutil
from pathlib import Path

def check_and_organize_dataset(data_dir):
    """
    Vérifie et compte les fichiers dans le dataset d'empreintes digitales.
    
    Args:
        data_dir (str): Chemin vers le dossier de données
    """
    # Groupes sanguins possibles
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    # Compteurs
    total_files = 0
    error_files = 0
    
    print("Vérification des fichiers dans le dataset...")
    
    # Parcours des dossiers de groupes sanguins
    for group in blood_groups:
        group_dir = os.path.join(data_dir, group)
        
        # Vérification si le dossier existe
        if not os.path.exists(group_dir):
            print(f"Attention: Le dossier {group} n'existe pas!")
            continue
            
        # Comptage des fichiers dans le dossier
        try:
            files = [f for f in os.listdir(group_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            count = len(files)
            total_files += count
            print(f"{group}: {count} fichiers")
            
            # Vérification de la taille des images
            for file in files:
                try:
                    file_path = os.path.join(group_dir, file)
                    if os.path.getsize(file_path) == 0:
                        print(f"Attention: Le fichier {file} est vide!")
                        error_files += 1
                except Exception as e:
                    print(f"Erreur lors de la vérification de {file}: {str(e)}")
                    error_files += 1
                    
        except Exception as e:
            print(f"Erreur lors du traitement du dossier {group}: {str(e)}")
            error_files += 1
    
    # Affichage des statistiques
    print("\nStatistiques du dataset:")
    print(f"Total des fichiers: {total_files}")
    print(f"Fichiers en erreur: {error_files}")
    
    # Vérification de la distribution des données
    if total_files > 0:
        print("\nDistribution des données:")
        for group in blood_groups:
            group_dir = os.path.join(data_dir, group)
            if os.path.exists(group_dir):
                count = len([f for f in os.listdir(group_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                percentage = (count / total_files) * 100
                print(f"{group}: {count} fichiers ({percentage:.1f}%)")
    else:
        print("\nAucun fichier trouvé dans le dataset!")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    check_and_organize_dataset(data_dir) 