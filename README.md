# Sign Language Recognition

Un projet d'apprentissage automatique pour reconnaître et classifier les gestes du langage des signes en temps réel.

## Description

Ce projet utilise **MediaPipe** pour détecter les landmarks (points clés) des mains et **scikit-learn** pour entraîner un classifieur afin de reconnaître différents gestes du langage des signes.

### Fonctionnalités
-  Capture d'images de gestes via webcam
-  Extraction des landmarks des mains avec MediaPipe
-  Création d'un dataset annoté
-  Entraînement d'un classifieur ML
-  Inférence en temps réel sur des images

##  Installation

### Prérequis
- Python 3.8+
- Webcam connectée
- Windows/Linux/Mac

### Étapes

1. **Cloner le repository**
   ```bash
   git clone <votre-repo>
   cd SignLanguage
   ```

2. **Créer un environnement virtuel**
   ```powershell
   python -m venv .venv
   ```

3. **Activer l'environnement virtuel**
   ```powershell
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

##  Utilisation

### 1. Collecter les données d'entraînement
```powershell
python collect_imgs.py
```
- Capture 100 images par classe (ajustable dans le code)
- Les images sont sauvegardées dans `data/0/`, `data/1/`, etc.
- Appuyez sur **Q** quand vous êtes prêt à commencer
- Placez votre main face à la caméra pour capturer les gestes

### 2. Créer le dataset
```powershell
python create_dataset.py
```
- Extrait les landmarks (points clés) des mains
- Crée des fichiers pickle (`data.pickle` et `labels.pickle`)
- Normalise les données pour l'entraînement

### 3. Entraîner le classifieur
```powershell
python train_classifier.py
```
- Entraîne un Random Forest sur les landmarks extraits
- Sauvegarde le modèle dans `model.p`
- Affiche les métriques d'entraînement

### 4. Faire de l'inférence
```powershell
python inference_classifier.py
```
- Lance la caméra en temps réel
- Affiche la prédiction du classifieur
- Montre les landmarks détectés


##  Dépendances

- **opencv-contrib-python** : Traitement d'images et capture vidéo
- **mediapipe** : Détection des landmarks des mains
- **scikit-learn** : Algorithme de classification (Random Forest)
- **numpy** : Calcul numérique
- **matplotlib** : Visualisations
- **pillow** : Traitement d'images supplémentaires
- **sounddevice** : Gestion audio (optionnel)



