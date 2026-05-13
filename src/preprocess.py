import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image

def load_dataset(data_dir, img_size=64):
    images, labels = [], []

    for label, category in enumerate(['Uninfected', 'Parasitized']):
        folder = os.path.join(data_dir, category)

        print(f"Loading from: {folder}")

        for fname in os.listdir(folder):
            if fname.endswith('.png') or fname.endswith('.jpg'):

                path = os.path.join(folder, fname)

                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize((img_size, img_size))

                    images.append(np.array(img) / 255.0)
                    labels.append(label)

                except Exception as e:
                    print(f"Error loading {path}: {e}")

    return np.array(images), np.array(labels)

X, y = load_dataset('data/cell_images/cell_images')

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)

# Save processed data
np.save('data/X_train.npy', X_train)
np.save('data/X_val.npy', X_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
np.save('data/y_test.npy', y_test)

print("Data saved successfully.")