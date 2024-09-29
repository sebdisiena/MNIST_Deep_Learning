# Deep Learning for MNIST Dataset Classification

This project implements a Neural Network to classify the MNIST dataset using Keras. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). This README provides an overview of the code, the approach taken to build the model, and the steps involved in loading, preprocessing, and evaluating the model.

## Table of Contents
- [Deep Learning for MNIST Dataset Classification](#deep-learning-for-mnist-dataset-classification)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset Overview](#dataset-overview)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Testing with External Images](#testing-with-external-images)
  - [Results](#results)

## Installation

1. Clone the repository and navigate to the directory.
2. Install the required packages using `pip`:
   ```bash
   pip install numpy matplotlib keras tensorflow requests pillow opencv-python
   ```

## Dataset Overview

The MNIST dataset contains 70,000 images of handwritten digits (60,000 for training and 10,000 for testing). Each image is 28x28 pixels, with pixel intensity values ranging from 0 to 255. These values are normalized for training.

The dataset is divided into:
- `X_train`: Training images.
- `y_train`: Labels for training images.
- `X_test`: Testing images.
- `y_test`: Labels for testing images.

The code loads and preprocesses the data by reshaping the images into a single row of 784 pixels, normalizing pixel values, and one-hot encoding the labels.

## Model Architecture

The model is a simple fully connected neural network built using Keras' Sequential API. It contains the following layers:

1. **Input Layer**: Takes the reshaped 784-pixel input.
2. **Three Hidden Layers**: Each with 10 units and ReLU activation.
3. **Output Layer**: Uses softmax activation to classify the image into one of 10 possible classes (digits 0-9).

```python
model = Sequential()
model.add(Dense(10, input_dim=784, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

The model is compiled using the Adam optimizer, categorical cross-entropy as the loss function, and accuracy as the evaluation metric.

## Training and Evaluation

The model is trained using the `fit()` function on the training dataset for 10 epochs with a batch size of 200. A validation split of 10% is used to monitor the model's performance on unseen data during training.

Training includes:
- **Loss**: Categorical cross-entropy to measure how well the model fits the data.
- **Accuracy**: Measures how well the model classifies digits.

```python
history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=200, verbose=1)
```

After training, the model is evaluated on the test dataset, with the test accuracy and loss printed for comparison.

```python
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score/Loss:", score[0])
print("Test Accuracy:", score[1])
```

## Testing with External Images

You can test the model on external images by loading a test image, resizing it to 28x28 pixels, and normalizing it to the same format as the MNIST dataset. The code demonstrates loading an image, preprocessing it using OpenCV, and predicting the class.

```python
img_array = np.asarray(img)
resized = cv2.resize(img_array, (28, 28))
grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(grayscale)
image = image / 255
image = image.reshape(1, 784)
```

The prediction is made by the model, and the most probable class is printed.

```python
prediction = model.predict(image)
predicted_class = prediction.argmax(axis=-1)
print("Predicted class:", predicted_class[0])
```

## Results

The model achieves a test accuracy of over 90%, demonstrating its effectiveness in recognizing handwritten digits. The plots for training and validation loss/accuracy can help visualize the performance of the model over the epochs.

---

