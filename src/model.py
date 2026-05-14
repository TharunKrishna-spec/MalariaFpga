import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(64, 64, 3)):
    model = models.Sequential([
        # Block 1 — output: 64x64x16 → pool: 32x32x16
        layers.Conv2D(16, (3,3), activation='relu', padding='same', name='conv2d', input_shape=input_shape),
        layers.MaxPooling2D((2,2), name='max_pooling2d'),

        # Block 2 — output: 32x32x32 → pool: 16x16x32
        layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_1'),
        layers.MaxPooling2D((2,2), name='max_pooling2d_1'),

        # Block 3 — output: 16x16x32 → pool: 8x8x32
        layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_2'),
        layers.MaxPooling2D((2,2), name='max_pooling2d_2'),

        # Classifier — 8x8x32 = 2048 → Dense 32 → output 1
        layers.Flatten(name='flatten'),
        layers.Dense(16, activation='relu', name='dense'),
        layers.Dense(1, activation='sigmoid', name='dense_1'),
    ])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    #model.summary()