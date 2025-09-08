1. Contexte et Motivation 
La détermination du groupe sanguin est essentielle en médecine, notamment pour les 
transfusions, les greffes et les urgences. Les méthodes traditionnelles (tests sérologiques) sont 
coûteuses et longues. 
Ce projet explore une approche innovante utilisant l’apprentissage automatique pour prédire le 
groupe sanguin à partir des empreintes digitales, une solution non invasive, rapide et 
économique. Les empreintes digitales, uniques et riches en caractéristiques (minuties, boucles, 
arches), permettent une classification fiable. 
2. Description du Dataset 
Le dataset utilisé pour ce projet provient de Kaggle et est intitulé "Finger Print based Blood 
Group Dataset". 
Lien :https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/
 data 
Ce dataset contient des images d'empreintes digitales accompagnées des informations relatives 
aux groupes sanguins des individus. Voici une description des principales caractéristiques du 
dataset : 
• Images des empreintes digitales : Chaque enregistrement du dataset est constitué d'une 
image d'empreinte digitale. Ces images sont de haute qualité et sont utilisées pour 
extraire les caractéristiques spécifiques qui définissent chaque empreinte. 
• Groupes sanguins associés : Chaque image d'empreinte digitale est étiquetée avec le 
groupe sanguin correspondant, ce qui permet de créer un modèle de classification 
supervisé. Les groupes sanguins sont classés dans les catégories ABO (A, B, AB, O) et 
Rhésus (positif ou négatif). 
• Caractéristiques des empreintes digitales : Bien que les caractéristiques spécifiques ne 
soient pas extraites de manière explicite dans ce dataset, elles peuvent être dérivées des 
images à l'aide de techniques de traitement d'images et d'extraction de caractéristiques, 
telles que les minuties, les boucles, et autres caractéristiques de la texture et des motifs 
des empreintes. 
Le dataset contient plus de 500 images d'empreintes digitales pour chaque groupe sanguin 
spécifique. 
3. Objectifs du projet 
L'objectif principal de ce projet est de prédire le groupe sanguin d'un individu en fonction de son 
empreinte digitale à l'aide de modèles de machine learning. 
4. Méthodologie 
a. Prétraitement des images 
• Conversion des images d'empreintes digitales en une forme numérique qui peut être 
traitée par les algorithmes de machine learning. 
• Normalisation et redimensionnement des images pour assurer une cohérence des 
dimensions. 
b. Extraction des caractéristiques 
• Utilisation de techniques de traitement d'image pour extraire des caractéristiques 
importantes telles que les minuties, les boucles et les arches des empreintes 
digitales. 
c. Entraînement des modèles 
• K-Nearest Neighbors (KNN) : Un algorithme de classification basé sur la proximité 
des exemples dans l'espace des caractéristiques. Il sera utilisé pour prédire le groupe 
sanguin en fonction des k voisins les plus proches. 
• Support Vector Machine (SVM) : Un autre algorithme de classification qui trouve une 
hyperplan pour séparer les différentes classes (groupes sanguins) dans l'espace des 
caractéristiques. 
• Random Forest (RF) : Un modèle d'ensemble qui utilise plusieurs arbres de décision 
pour améliorer la précision de la classification. 
• Arbres de Décision (DT) : Structure arborescente où chaque nœud représente un test 
sur une caractéristique, chaque branche un résultat et chaque feuille une classe 
(groupe sanguin). 
d. Évaluation 
• Utilisation des métriques de performance telles que la précision, le rappel et le 
F1-score pour évaluer la qualité des prédictions effectuées par les modèles.
