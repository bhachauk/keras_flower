import keras_flower as kf

file_name = "example/sonflower.jpeg"

predicted, score = kf.predict_name_by_path(file_name)[0]

assert predicted == 'sunflower'

assert score >= 0.7
