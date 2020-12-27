## Keras Flower

#### Training data

- [Petals to the Metal - Flower Classification on TPU](https://www.kaggle.com/c/tpu-getting-started)


#### Results

![demo](demo.gif)


#### Usage

- To get all prediction results

```
import keras_flower as kf
predictions = kf.predict("file/to/predict.png")
```


- To get top prediction result with flower labels

```
import keras_flower as kf
for predicted, score in kf.predict_name("/path/to/file.png"):
    print(predicted, score)
```