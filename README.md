# README

This Python script provides the implementation for an image classification task using various models. It contains a training and testing pipeline for loading and processing the dataset, building and training the model, and evaluating the performance on a test set.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib

## Usage

To run the script, use the following command:

```bash
python <guideline.py> --phase <train/test> [options]
```
## Options
--phase: Choose between 'train' and 'test'. 'train' will train the model, and 'test' will test the pre-trained model.

--train_data_dir: The directory of the training data (default: ./data/train/).

--test_data_dir: The directory of the testing data (default: ./data/test/).

--model_dir: The directory of the saved model (default: model.pth).

--train_val_split: Percent of train data for train/validation split (default: 0.8).

--use_transforms: Use image transforms during preprocessing (default: True).

--model: Choose the model architecture: 'cnn', 'resnet', or 'resnet50' (default: 'cnn').

--input_size: Input size of the model (default: (224, 224)).

--cls_nums: Number of classes (default: 15).

--use_gpu: Use GPU for training and testing (default: True).

--batch_size: Batch size for training and testing (default: 32).

--lr: Learning rate for the optimizer (default: 1e-3).

--momentum: Momentum for the SGD optimizer (default: 0.9).

--weight_decay: L2 regularization (default: 5e-4).

--lr_decay_step: Step size for learning rate decay (default: 40).

--lr_decay_factor: Learning rate decay factor (default: 0.8).

--num_epochs: Number of epochs for training (default: 200).

## Example

Here are some example commands to run the script for training and testing using the ResNet50 model:

### Training
```bash
python guideline.py --phase train --model resnet50 --lr 0.001 --num_epochs 100
```

### Testing
```bash
python guideline.py --phase test --model resnet50
```
## EarlyStopping Note
Please note that the provided EarlyStopping functionality is just used for ResNet50, but not used for CNN and ResNet models. If you choose to use a CNN or ResNet model, you should comment out the EarlyStopping part in the script. To do so, locate the following lines of code:

```python
# early_stopping = EarlyStopping(patience=patience, verbose=True)
# ...
# early_stopping(valid_loss, model)
# ...
# if early_stopping.early_stop:
#     print("Early stopping")
#     break
