# <center> Image Classification with PyTorch </center>
<p>
  <img src="https://img.shields.io/badge/matplotlib-3.7.0-green">
  <img src="https://img.shields.io/badge/numpy-1.24.2-green">
  <img src="https://img.shields.io/badge/pandas-1.5.3-green">
  <img src="https://img.shields.io/badge/Pillow-9.4.0-green">
  <img src="https://img.shields.io/badge/scikit--learn-1.2.1-green">
  <img src="https://img.shields.io/badge/scipy-1.10.0-green">
  <img src="https://img.shields.io/badge/torch-1.13.1-green">
  <img src="https://img.shields.io/badge/torchvision-0.14.1-green">

![Image Classification using PyTorch](https://miro.medium.com/v2/resize:fit:679/1*bhFifratH9DjKqMBTeQG5A.gif)

## Deep learning course project

Tackling image classification, a core aspect of Computer Vision, is the focus of this repository. Utilizing PyTorch, a popular framework, this project embraces transfer learning. This approach not only saves time and resources but often yields superior results compared to building and training a neural network from scratch. The repository features image classification solutions using various algorithms within the PyTorch ecosystem:
* EfficientNet
* ResNet
* VGG
* GoogLeNet

### Model Training Configuration

Before training, update the configuration file:

* Loss Function: `CrossEntropyLoss` is recommended for binary and multi-class classification. Choose between `CrossEntropyLoss` and `NLLLoss`.
* Optimization Function: Options include Adam, `RAdam`, `SGD`, `Adadelta`, `Adagrad`, `AdamW`, `Adamax`, `ASGD`, `NAdam`, and `Rprop`, with `Adam` being recommended.
* MODEL_NAME: Options are `efficientnetB0` to `efficientnetB7` for Efficientnet, `resnet18` to `resnet152` for Resnet, `vgg11` to `vgg19bn` for VGG, and `googlenet`.
* SAVE_WEIGHT_PATH: Directory to save model weights.
* DATA_DIR: Directory containing the dataset.
* CHECKPOINT: Directory for pretrained models.
* NUMCLASS: Number of classes.

Adjust other hyperparameters like EPOCHS, BATCHSIZE, and LEARNING_RATE as needed. To train:

```
cd ./src && python3 train.py
```

### Inference
Ensure that the model name, checkpoint, and number of classes in the config file match those used during training:

```
cd ./src && python predict.py \
        --test_path ../test_img \
        --batch_predict 16
```

* --test_path: Path to public test images (file or directory).
* --batch_predict: Batch size for prediction.

Results will be available in `predict.csv`.



