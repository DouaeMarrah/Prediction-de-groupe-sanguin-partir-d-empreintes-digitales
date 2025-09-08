import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from skimage.feature import local_binary_pattern, hog

class FingerprintPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détection d'Empreintes")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Charger les données prétraitées et le modèle
        self.preprocessed_data = self.load_preprocessed_data()
        self.model, self.scaler = self.load_model()
        
        # Variables
        self.image_path = None
        self.prediction = None
        self.probabilities = None
        
        # Style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#4CAF50")
        self.style.configure("TLabel", padding=6, background="#f0f0f0")
        
        # Créer l'interface
        self.create_widgets()
        
    def load_preprocessed_data(self):
        """Charge les données prétraitées."""
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'processed_data', 'preprocessed_dataset.pkl')
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les données prétraitées: {str(e)}")
            return None
    
    def load_model(self):
        """Charge le modèle entraîné."""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            model_files = [f for f in os.listdir(models_dir) if f.startswith('best_model_')]
            if not model_files:
                raise FileNotFoundError("Aucun modèle trouvé dans le dossier models")
            
            latest_model = max(model_files)
            model_path = os.path.join(models_dir, latest_model)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data['model'], model_data['scaler']
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle: {str(e)}")
            return None, None
    
    def create_widgets(self):
        """Crée les widgets de l'interface."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, 
                              text="Détection d'Empreintes Digitales",
                              font=("Helvetica", 16, "bold"))
        title_label.pack(pady=20)
        
        # Frame pour l'image et les contrôles
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour l'image
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        
        # Label pour l'image
        self.image_label = ttk.Label(image_frame, text="Aucune image sélectionnée")
        self.image_label.pack(pady=10)
        
        # Canvas pour afficher l'image
        self.canvas = tk.Canvas(image_frame, width=400, height=400, bg="white")
        self.canvas.pack(pady=10)
        
        # Frame pour les contrôles
        control_frame = ttk.Frame(content_frame)
        control_frame.pack(side=tk.RIGHT, padx=20, fill=tk.Y)
        
        # Bouton pour charger une image
        load_button = ttk.Button(control_frame, 
                               text="Charger une Empreinte",
                               command=self.load_image)
        load_button.pack(pady=20)
        
        # Bouton pour prédire
        predict_button = ttk.Button(control_frame,
                                  text="Analyser l'Empreinte",
                                  command=self.predict)
        predict_button.pack(pady=20)
        
        # Frame pour les résultats
        result_frame = ttk.Frame(control_frame)
        result_frame.pack(pady=20)
        
        # Label pour le résultat
        self.result_label = ttk.Label(result_frame, 
                                    text="Classe Prédite: -",
                                    font=("Helvetica", 12))
        self.result_label.pack(pady=10)
        
        # Label pour la confiance
        self.confidence_label = ttk.Label(result_frame,
                                        text="Confiance: -",
                                        font=("Helvetica", 12))
        self.confidence_label.pack(pady=10)
        
        # Canvas pour le graphique des probabilités
        self.prob_canvas = tk.Canvas(result_frame, width=300, height=200, bg="white")
        self.prob_canvas.pack(pady=10)
        
        # Footer
        footer_label = ttk.Label(main_frame,
                               text="© 2024 - Système de Détection d'Empreintes",
                               font=("Helvetica", 8))
        footer_label.pack(side=tk.BOTTOM, pady=10)
    
    def load_image(self):
        """Charge une image d'empreinte digitale."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.bmp *.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            # Afficher l'image
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Affiche l'image dans le canvas."""
        try:
            # Charger l'image
            image = Image.open(image_path)
            # Redimensionner l'image pour l'affichage
            image = image.resize((400, 400), Image.Resampling.LANCZOS)
            # Convertir en PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Mettre à jour le canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Garder une référence
            
            # Mettre à jour le label
            self.image_label.config(text=os.path.basename(image_path))
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {str(e)}")

    def preprocess_image(self, image_path):
        """Prétraite l'image pour la prédiction en utilisant les mêmes paramètres que le dataset."""
        if not self.preprocessed_data:
            raise ValueError("Les données prétraitées n'ont pas été chargées correctement")
            
        # Lecture de l'image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
            
        # Redimensionnement à la même taille que les images d'entraînement (128x128)
        target_size = (128, 128)  # Taille utilisée dans le dataset
        img = cv2.resize(img, target_size)
        
        # Normalisation comme dans le dataset
        img = img.astype('float32') / 255.0
        
        # Utiliser les mêmes paramètres que ceux du dataset
        features = []
        
        # 1. HOG features
        hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), visualize=False)
        features.extend(hog_features)
        
        # 2. LBP features
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        features.extend(hist_lbp)
        
        # 3. Statistiques basiques
        features.extend([
            np.mean(img),
            np.std(img),
            np.median(img),
            np.max(img),
            np.min(img)
        ])
        
        # Vérifier que nous avons le même nombre de caractéristiques que dans le dataset
        features = np.array(features)
        expected_features = self.preprocessed_data['X_train'].shape[1]
        if len(features) != expected_features:
            raise ValueError(f"Nombre de caractéristiques incorrect. Attendu: {expected_features}, Obtenu: {len(features)}")
            
        return features
    
    def draw_probabilities(self, probabilities):
        """Dessine le graphique des probabilités."""
        self.prob_canvas.delete("all")
        
        # Dimensions du canvas
        width = self.prob_canvas.winfo_width()
        height = self.prob_canvas.winfo_height()
        
        # Paramètres du graphique
        bar_width = width / len(probabilities)
        max_prob = max(probabilities)
        
        # Dessiner les barres
        for i, prob in enumerate(probabilities):
            x1 = i * bar_width
            y1 = height
            x2 = (i + 1) * bar_width
            y2 = height - (prob / max_prob) * height
            
            # Barre
            self.prob_canvas.create_rectangle(x1, y1, x2, y2, fill="blue")
            
            # Valeur
            self.prob_canvas.create_text(x1 + bar_width/2, y2 - 10,
                                       text=f"{prob:.2f}",
                                       font=("Helvetica", 8))
            
            # Groupe sanguin
            blood_group = self.preprocessed_data['classes'][i]
            self.prob_canvas.create_text(x1 + bar_width/2, height - 10,
                                       text=blood_group,
                                       font=("Helvetica", 8))
    
    def predict(self):
        """Prédit la classe de l'empreinte."""
        if not self.image_path:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une image")
            return
        
        if not self.model or not self.preprocessed_data:
            messagebox.showerror("Erreur", "Modèle ou données prétraitées non chargés")
            return
        
        try:
            # Prétraiter l'image
            features = self.preprocess_image(self.image_path)
            
            # Normaliser les caractéristiques
            features_scaled = self.scaler.transform([features])
            
            # Faire la prédiction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Obtenir le groupe sanguin à partir des classes du dataset
            blood_group = self.preprocessed_data['classes'][prediction]
            
            # Mettre à jour l'interface
            self.result_label.config(
                text=f"Groupe Sanguin Prédit: {blood_group}"
            )
            self.confidence_label.config(
                text=f"Confiance: {probabilities[prediction]:.2%}"
            )
            
            # Dessiner le graphique des probabilités
            self.draw_probabilities(probabilities)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintPredictorApp(root)
    root.mainloop() 