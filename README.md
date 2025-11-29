# fashion_mnist_cnn_classifier
CNN image classifier for Fashion MNIST using Keras on CSV-formatted data.
# Fashion MNIST – CNN Image Classifier

This project implements a convolutional neural network (CNN) to classify images from the Fashion MNIST dataset into ten clothing categories. The dataset is used in CSV format (e.g. `fashion-mnist_train.csv` and `fashion-mnist_test.csv`), where each row represents one image.

The code loads the CSV files, reshapes the pixel values into 28×28 grayscale images, trains a CNN using TensorFlow/Keras, and generates a submission file with predicted labels.

---

## 1. Techniques and Tools

- Language: Python  
- Libraries: `tensorflow`, `keras`, `numpy`, `pandas`  
- Methods:
  - Convolutional neural networks (CNNs)
  - Batch normalisation
  - Dropout regularisation
  - Training/validation split

---

## 2. Repository Structure

- `fashion_mnist_cnn.py`  
  - Loads training and test data from CSV  
  - Normalises and reshapes image data to `(28, 28, 1)`  
  - Builds and trains a CNN model  
  - Evaluates the model on the training set  
  - Writes a `submission.csv` file with predicted labels

(Optional) The original notebook used for experimentation can be added under a `notebooks/` directory.

---

## 3. How to Run

1. Place the CSV files in a `data/` directory (or update the paths in the script):

   - `data/fashion-mnist_train.csv`  
   - `data/fashion-mnist_test.csv`

2. Install dependencies:

   ```bash
   pip install tensorflow numpy pandas
