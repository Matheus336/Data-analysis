# Importando as bibliotecas os arquivos necessários

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/wine+quality/winequality-red.csv'
import os
if os.path.exists(file_path):
    print("File exists!")
    try:
      print("Reading file...")
      data = pd.read_csv(file_path, sep=';')
    except:
      print('Error')
    finally:
      print('Done!')
else:
    print("File does not exist")


# Aqui estou separando os atributos das labels e dividindo a base de dados em treino e teste seguindo o critério específicado, 80% para treinamento e 20% para teste.

try:
  print('Checking column "quality"...')
  X = data.drop('quality', axis=1)
except:
   print('Column "quality" not found in DataFrame')
finally:
   print('Done!')

y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Treinando o modelo Random Forest
try:
  print('Training model using Random Forest Classifier...')
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
except:
   print('Error when training model')
finally:
   print('Done!')


# Fazendo previsões e visualizando resultados

try:
  print('Making predictions and calculating accuracy...')
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)
except:
   print('Error when calculating accuracy')
finally:
   print('Done!\n\n')


print('================================   RESULTS RANDOM FOREST  ================================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo de redes neurais

try:
    print('Training model using MLPClassifier...')
    model = MLPClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('================================   RESULTS NEURAL NETWORK  ================================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo de Naive Bayes

try:
    print('Training model using Naive Bayes...')
    model = GaussianNB()
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   NAIVE BAYES RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo KNeighbors

try:
    print('Training model using KNeighborsClassifier...')
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   KNEIGHBORS RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo Máquinas de Vetores de Suporte

try:
    print('Training model using SVM...')
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   SVM RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo de árvores de decisão

try:
    print('Training model using Decision Tree...')
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   DECISION TREE RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo Processo Gaussiano

try:
    print('Training model using Gaussian Process Classifier...')
    model = GaussianProcessClassifier()
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   GAUSSIAN PROCESS CLASSIFIER RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


# Treinando o modelo Gradient Boosting

try:
    print('Training model using Gradient Boosting Classifier...')
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
except:
    print('Error when training model')
finally:
    print('Done!')


# Fazendo previsões e visualizando resultados

try:
    print('Making predictions and calculating accuracy...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except:
    print('Error when calculating accuracy')
finally:
    print('Done!\n\n')

print('===============================   GRADIENT BOOSTING CLASSIFIER RESULTS  ===============================\n\n')
print("Accuracy: %.2f%%\n\n" % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print('\n\n')


