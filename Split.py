# Bibliotecas básicas
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image

# TensorFlow e tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.utils import to_categorical

# Aprendizado de Máquina
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Funções personalizadas
from functions import load_images, create_model, train_model

#Recolha das Imagens Que Serão Utilizadas
NOImage = '/TPdeRedesNeurais/no'
YESImage = '/TPdeRedesNeurais/yes'

#Carregamento das Imagens
arrayNOImage, NOLabel = load_images(NOImage)
arrayYESImage, YESLabel = load_images(YESImage)

#Concatenação das Imagens e Labels
All_Images = arrayYESImage + arrayNOImage
All_label = YESLabel + NOLabel

#Converte Lista de Arrays e Labels em arrays NumPy
X = np.array(All_Images)
Y = np.array(All_label)

# Número de iterações desejadas
num_iterations = 10
accuracy_results = []

for i in range(num_iterations):
    # Definindo a estratégia de train_test_split com 80% train e 20% teste
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)

    # Criando modelo
    model = create_model()

    model.summary()
    
    # Treinando o modelo com os dados de treino
    history = train_model(model, x_train, y_train, epochs=70, validation_split=0.1)

    # Avaliando a acurácia do modelo nos dados de teste
    accuracy = model.evaluate(x_test, y_test)[1]
    accuracy_results.append(accuracy)

    # Obter as previsões do modelo
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Converte os rótulos para one-hot encoding
    y_test_encoded = to_categorical(y_test)
    y_pred_labels_encoded = to_categorical(y_pred_labels)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(y_test_encoded.argmax(axis=1), y_pred_labels_encoded.argmax(axis=1))

    # Plotar a matriz de confusão usando seaborn
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 25})
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Não", "Sim"], yticklabels=["Não", "Sim"])
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")

    # Salvar a figura como um arquivo PNG
    plt.savefig("matriz_confusao_iteracao_{}.png".format(i+1), dpi=600)

    # Exibir a figura
    plt.show()

# Calcular o desvio padrão das precisões
std_deviation = np.std(accuracy_results)

print(f"Resultados de Acurácia em {num_iterations} iterações: {accuracy_results}")
average_accuracy = np.mean(accuracy_results)
print(f"Média das Acurácias: {average_accuracy}")
print(f"Desvio Padrão das Acurácias: {std_deviation}")