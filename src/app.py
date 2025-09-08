import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from skimage.feature import local_binary_pattern, hog

class BloodGroupPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction de Groupe Sanguin")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Charger le modèle
        self.model = self.load_model()
        
        # Variables
        self.image_path = None
        self.prediction = None
        self.confidence = None
        
        # Style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#4CAF50")
        self.style.configure("TLabel", padding=6, background="#f0f0f0")
        
        # Créer l'interface
        self.create_widgets()
        
    def load_model(self):
        """Charge le modèle entraîné."""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'models', 'best_model.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle: {str(e)}")
            return None
    
    def create_widgets(self):
        """Crée les widgets de l'interface."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, 
                              text="Prédiction de Groupe Sanguin par Empreinte Digitale",
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
                                  text="Prédire le Groupe Sanguin",
                                  command=self.predict)
        predict_button.pack(pady=20)
        
        # Frame pour les résultats
        result_frame = ttk.Frame(control_frame)
        result_frame.pack(pady=20)
        
        # Label pour le résultat
        self.result_label = ttk.Label(result_frame, 
                                    text="Groupe Sanguin Prédit: -",
                                    font=("Helvetica", 12))
        self.result_label.pack(pady=10)
        
        # Label pour la confiance
        self.confidence_label = ttk.Label(result_frame,
                                        text="Confiance: -",
                                        font=("Helvetica", 12))
        self.confidence_label.pack(pady=10)
        
        # Footer
        footer_label = ttk.Label(main_frame,
                               text="© 2024 - Système de Prédiction de Groupe Sanguin",
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

    def extract_features(self, image):
        """Extrait les caractéristiques de l'image."""
        features = []
        
        # 1. HOG features
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), visualize=False)
        features.extend(hog_features)
        
        # 2. LBP features
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        features.extend(hist_lbp)
        
        # 3. Basic statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.median(image),
            np.max(image),
            np.min(image)
        ])
        
        return np.array(features)
    
    def predict(self):
        """Prédit le groupe sanguin à partir de l'image."""
        if not self.image_path:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une image")
            return
        
        if not self.model:
            messagebox.showerror("Erreur", "Modèle non chargé")
            return
        
        try:
            # Prétraiter l'image
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Même taille que pendant l'entraînement
            img = img.astype('float32') / 255.0
            
            # Extraire les caractéristiques
            features = self.extract_features(img)
            
            # Normaliser les caractéristiques
            features_scaled = self.model['scaler'].transform(features.reshape(1, -1))
            
            # Faire la prédiction
            prediction = self.model['model'].predict(features_scaled)[0]
            
            # Essayer d'obtenir la probabilité si disponible
            try:
                proba = self.model['model'].predict_proba(features_scaled)[0]
                confidence = np.max(proba)
                confidence_text = f"{confidence:.2%}"
            except:
                confidence_text = "Non disponible"
            
            # Mettre à jour l'interface
            self.result_label.config(
                text=f"Groupe Sanguin Prédit: {self.model['classes'][prediction]}"
            )
            self.confidence_label.config(
                text=f"Confiance: {confidence_text}"
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BloodGroupPredictorApp(root)
    root.mainloop() 