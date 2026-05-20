import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#script de reconnaissance de langue des signes en temps réel

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# shuffle: Mélange les données aléatoirement avant séparation
#20% des données sont utilisées pour les tests, et 80% pour l'entraînement
#stratify: Conserve les proportions des classes.
# x_train et x_test: Données utilisées pour entraîner le modèle.
# y_train et y_test: Labels correspondants à x_train et x_test, respectivement.
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

#sauvegarder le modèle entrainé dans un fichier nommé model.p
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()