import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_input_shape = (224, 224)
# model = tf.keras.models.load_model(os.path.abspath("keras_flower.h5"))
model = tf.keras.Sequential([
        tf.keras.applications.DenseNet201(weights=None, include_top=False, input_shape=[*model_input_shape, 3]),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(108, activation='softmax')
    ])
model.load_weights(os.path.abspath("keras_flower_weights.h5"))
labels = np.loadtxt(os.path.abspath("keras_flower/labels.txt"), dtype='str', delimiter="\n")


def predict(act_path):
    img = Image.open(act_path).resize(model_input_shape)
    image = np.array(img)
    image = image / 255.0
    results = model.predict(np.expand_dims(image, axis=0))[0]
    return results


def predict_name(act_path, top=1):
    results = predict(act_path)
    return sorted(zip(labels, results), key=lambda x: x[1], reverse=True)[:top]
