import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Création du dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chargement du meilleur modèle
def load_best_model():
    models_dir = 'models'
    # Trouver le fichier du meilleur modèle le plus récent
    model_files = [f for f in os.listdir(models_dir) if f.startswith('best_model_')]
    if not model_files:
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier models")
    
    latest_model = max(model_files)
    model_path = os.path.join(models_dir, latest_model)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['scaler']

# Vérification de l'extension du fichier
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prétraitement de l'image
def preprocess_image(image_path):
    # Lecture de l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Redimensionnement si nécessaire
    img = cv2.resize(img, (64, 64))
    
    # Normalisation
    img = img / 255.0
    
    # Aplatir l'image
    features = img.flatten()
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Sauvegarde du fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Chargement du modèle
            model, scaler = load_best_model()
            
            # Prétraitement de l'image
            features = preprocess_image(filepath)
            
            # Normalisation des caractéristiques
            features_scaled = scaler.transform([features])
            
            # Prédiction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Création de la visualisation
            plt.figure(figsize=(10, 5))
            
            # Affichage de l'image originale
            plt.subplot(1, 2, 1)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title('Image Originale')
            plt.axis('off')
            
            # Affichage des probabilités
            plt.subplot(1, 2, 2)
            classes = model.classes_
            plt.bar(classes, probabilities)
            plt.title('Probabilités de Classification')
            plt.xlabel('Classes')
            plt.ylabel('Probabilité')
            plt.xticks(rotation=45)
            
            # Sauvegarde du graphique en mémoire
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # Nettoyage
            os.remove(filepath)
            
            return jsonify({
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'plot': plot_data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

if __name__ == '__main__':
    app.run(debug=True) 