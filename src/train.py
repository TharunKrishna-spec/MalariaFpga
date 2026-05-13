import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_model

# Load saved data
X_train = np.load('data/X_train.npy')
X_val   = np.load('data/X_val.npy')
y_train = np.load('data/y_train.npy')
y_val   = np.load('data/y_val.npy')

model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('model/best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Save final model
model.save('model/malaria_cnn.keras')
print("Training complete. Model saved.")