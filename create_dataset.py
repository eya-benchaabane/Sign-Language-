import os
import pickle

import mediapipe as mp #de détecter, suivre et analyser des éléments dans une image
import cv2  #capture l’image et la traite
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands #Détection et suivi des mains
mp_drawing = mp.solutions.drawing_utils #Dessiner landmarks et lignes
mp_drawing_styles = mp.solutions.drawing_styles #Ajouter des styles visuels

#Création du détecteur de mains
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = [] #liste de listes, chaque liste interne contient les coordonnées des landmarks d’une image x et y
labels = [] #catégorie de chaque image, par exemple 0, 1, 2 pour les classes de signes
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #OpenCV lit les images en BGR mais MediaPipe travaille en RGB donc on convertit les couleurs.

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    #Normalisation des coordonnées. On rend les coordonnées indépendantes de la position de la main (gauche/droite) et de la taille de la main.
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux) # utile pour entrainer le classificateur 
            labels.append(dir_)

#crée et ouvre un fichier nommé data.pickle pour sauvegarder les données sous forme binaire
f = open('data.pickle', 'wb')
#sauvegarde un dictionnaire Python dans le fichier. 
#Le dictionnaire contient deux clés : 'data' et 'labels', associées respectivement aux listes data et labels.
pickle.dump({'data': data, 'labels': labels}, f)
f.close()