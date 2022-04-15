import keras_flower as kf
import numpy as np
from PIL import Image


def test_sunflower():
    file_name = "example/sonflower.jpeg"
    predicted, score = kf.predict_name_by_path(file_name)[0]
    assert predicted == 'sunflower'
    assert score >= 0.7


def test_sunflower_array():
    file_name = "example/sonflower.jpeg"
    image = Image.open(file_name).resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def test_embed():
    embed = kf.embed_by_path("example/sonflower.jpeg")
    assert 1920 == embed.shape[0]


if __name__ == '__main__':
    # embed = kf.embed_by_path("example/sonflower.jpeg")
    # print(embed)
    # print(embed.shape)
    # file_name = "example/sonflower.jpeg"
    # predicted, score = kf.predict_name_by_path(file_name)[0]
    # print(predicted)
    pass
