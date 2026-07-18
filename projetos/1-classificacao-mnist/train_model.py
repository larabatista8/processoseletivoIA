import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization
import numpy as np

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui
#1. Carregar o dataset MNIST via tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#model.fit(x_train, y_train, batch_size=32, validation_split=0.2,epochs=1)

# 4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)

model = Sequential([
        # Bloco Convolucional 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco Convolucional 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco Convolucional 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Transforma a matriz 2D em um vetor 1D
        Flatten(),
        
        # Dropout antes da camada de saída para evitar overfitting
        Dropout(0.5),
        
        # Camada de Saída (10 classes)
        Dense(10, activation='softmax')
    ])
    
    #   5. Treinar com EarlyStopping monitorando a perda de validação