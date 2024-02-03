import tensorflow as tf
import cv2
import os
from tensorflow.keras import layers, models

def train_model(model, x_train, y_train, epochs=70, validation_split=0.1):
    # Configuração do agendamento de taxa de aprendizado exponencial
    initial_learning_rate = 0.001
    total_steps = 25 * len(x_train) // 32
    decay_steps = total_steps
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=False)
    
    #Configuração do otimizador Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # Compilação do modelo com a configuração do otimizador, função de perda e métricas
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Treinamento do modelo usando os dados de treino e validação
    history = model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs)
    return history

def create_model():
    # Carrega o modelo EfficientNetB0 pré-treinado
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3))

    # Torna as camadas do modelo base treináveis
    base_model.trainable = True
    
    # Define a entrada do modelo
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Normalização dos valores entre [0,1]
    rescaling_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(inputs)

    # Passa as imagens de entrada pelo modelo EfficientNetB0 pré-treinado
    new_model = base_model(rescaling_layer, training=True)

    #Camadas Adicionadas no Modelo
    x = tf.keras.layers.Flatten()(new_model)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    #Camada de dropout pra evitar overfitting
    x = tf.keras.layers.Dropout(0.3)(x)

    #Camadas Densas Adicionadas no Modelo
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Camada de saída com ativação softmax para classificação multiclasse (4 classes)
    prediction_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

    # Cria o modelo final com entrada e camada de saída
    model = tf.keras.Model(inputs, prediction_layer)
    return model

def load_images(directory):
    # Lista para armazenar as imagens e os labels correspondentes
    array_images = []
    labels = []

    # Iteração sobre os arquivos no diretório fornecido
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Caminho completo para a imagem
            image_path = os.path.join(directory, filename)

            # Lê a imagem usando OpenCV
            image = cv2.imread(image_path)

            # Redimensiona a imagem para o tamanho desejado (224x224 pixels)
            resized_image = cv2.resize(image, (224, 224))

            # Adiciona a imagem redimensionada à lista de imagens
            array_images.append(resized_image)

            # Adiciona o rótulo à lista de rótulos (0 para "no" e 1 para "yes" no nome do arquivo)
            labels.append(0 if "no" in filename else 1)
    
    # Retorna a lista de imagens e os rótulos correspondentes
    return array_images, labels