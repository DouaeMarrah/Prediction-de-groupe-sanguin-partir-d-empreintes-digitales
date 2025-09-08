#  Blood Group Prediction from Fingerprints  

##  1. Contexte et Motivation  
La détermination du groupe sanguin est une étape essentielle en médecine, notamment pour :  
- Les transfusions,  
- Les greffes,  
- Les situations d’urgence.  

Les méthodes traditionnelles (tests sérologiques) sont souvent **coûteuses et chronophages**.  
Ce projet explore une approche **innovante et non invasive** : prédire le groupe sanguin à partir des **empreintes digitales** en utilisant l’**apprentissage automatique**.  

Les empreintes digitales, uniques et riches en caractéristiques (minuties, boucles, arches), peuvent offrir une **solution rapide, économique et fiable**.  

---

## 2. Dataset  
Le dataset utilisé provient de Kaggle à telecharger avant :  
 [Finger Print based Blood Group Dataset](https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/)  

### Contenu du dataset :  
- **Images d’empreintes digitales** : données brutes servant à l’extraction des caractéristiques.  
- **Groupes sanguins associés** : chaque image est étiquetée selon le système **ABO (A, B, AB, O)** et le facteur **Rhésus (+ ou -)**.  
- **Taille** : plus de **500 images par groupe sanguin**.  

Les caractéristiques biométriques (minuties, arches, boucles, texture) peuvent être extraites via des techniques de **traitement d’image**.  

---

##  3. Objectifs du Projet  
- Développer un modèle de **machine learning** capable de prédire le groupe sanguin à partir d’une empreinte digitale.  
- Comparer différents algorithmes afin d’identifier le plus performant.  

---

##  4. Méthodologie  

### a. Prétraitement des images  
- Conversion en format exploitable par les modèles.  
- Normalisation et redimensionnement pour homogénéiser les données.  

### b. Extraction des caractéristiques  
- Utilisation de **techniques de traitement d’images** pour extraire les minuties, arches, boucles et textures.  

### c. Entraînement des modèles  
Plusieurs algorithmes supervisés sont testés :  
- **K-Nearest Neighbors (KNN)** : classification basée sur la proximité des voisins.  
- **Support Vector Machine (SVM)** : séparation optimale des classes via un hyperplan.  
- **Random Forest (RF)** : ensemble d’arbres de décision pour améliorer la robustesse.  
- **Decision Tree (DT)** : classification hiérarchique basée sur des règles simples.  

### d. Évaluation  
Les performances des modèles sont mesurées à l’aide de métriques classiques :  
- **Précision (Accuracy)**  
- **Rappel (Recall)**  
- **F1-score**  

---

##  5. Technologies Utilisées  
- **Python**  
- Bibliothèques principales :  
  - `numpy`, `pandas` → manipulation des données  
  - `opencv`, `scikit-image` → traitement d’images  
  - `scikit-learn` → algorithmes de machine learning  
  - `matplotlib`, `seaborn` → visualisation des résultats  

---
