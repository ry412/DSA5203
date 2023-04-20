"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Zheng Huan <huan_zheng@u.nus.edu>
"""
# Import necessary libraries
import os
import json
import random
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision.models import _utils

from dataset import dataset
from model.Resnet import BasicBlockModel as ResnetModel
from model.BasicCNN import Model as CNNModel

# Load class information from the JSON file
with open("classes.json") as f:
    classes = json.load(f)

# Set random seeds for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

# Ignore certain warnings
warnings.filterwarnings("ignore", category=UserWarning, module=_utils.__name__)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

###################################### Subroutines #####################################################################
# The following subroutines are examples that you might need for your implementation.
# You can add, modify, or delete these subroutines as needed.

def build_vocabulary(**kwargs):
    pass

def get_hist(**kwargs):
    pass

def classifier(**kwargs):
    pass

def get_accuracy(**kwargs):
    pass

def save_model(model, model_name):
    """
    Save the model to the specified directory.

    Args:
        model (nn.Module): The model to save.
        model_name (str): The name of the model.
    """
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, os.path.join(model_dir, 'model.pth'))

class EarlyStopping:
    """
    Early stopping to stop the training when the validation loss does not improve after a certain number of epochs.
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.best_model = model.state_dict()  # Save the best model state dictionary
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return False

    @staticmethod
    def save_checkpoint(val_loss, model):
        """
        Save the model if the validation loss has decreased.

        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The model to save.
        """
        print(f'Validation loss decreased ({val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'best_model.pt')
def split_train_val_data(train_data_dir, train_val_split):
    """
    Split the dataset into train and validation sets based on the given train_val_split ratio.

    Args:
        train_data_dir (str): The directory containing the training data.
        train_val_split (float): The ratio of training data to total data.

    Returns:
        train_images (list): A list of filepaths for the training images.
        train_labels (list): A list of labels for the training images.
        val_images (list): A list of filepaths for the validation images.
        val_labels (list): A list of labels for the validation images.
    """
    print("Split data...")
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    # Iterate through each class
    for i in range(len(classes)):
        sub_class = classes[str(i + 1)]
        sub_dir = os.path.join(train_data_dir, sub_class)
        image_names = os.listdir(sub_dir)

        # Calculate the number of training images based on train_val_split
        train_num = int(len(image_names) * train_val_split)

        # Randomly shuffle the image names to ensure a random distribution of images
        random.shuffle(image_names)

        # Split the image names into training and validation sets
        for name in image_names[:train_num]:
            train_images.append(os.path.join(sub_dir, name))
            train_labels.append(i)
        for name in image_names[train_num:]:
            val_images.append(os.path.join(sub_dir, name))
            val_labels.append(i)

    print("Finished")
    return train_images, train_labels, val_images, val_labels



###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""


def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    # Device configuration
    use_gpu = kwargs["extra_args"]["use_gpu"]
    if use_gpu:
        use_gpu = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if use_gpu else "cpu")
    print(f"device = {device}")

    # Split the dataset into training and validation sets
    train_images, train_labels, val_images, val_labels = split_train_val_data(opt.train_data_dir, opt.train_val_split)
    train_num, val_num = len(train_labels), len(val_labels)
    print(f"train_num = {train_num}, val_num = {val_num}")

    # Create the training and validation datasets
    train_set = dataset(train_images, train_labels, 'train',
                        kwargs["extra_args"]["use_transforms"], kwargs["extra_args"]["input_size"])
    val_set = dataset(val_images, val_labels, 'val',
                      kwargs["extra_args"]["use_transforms"], kwargs["extra_args"]["input_size"])

    # Create data loaders for the training and validation sets
    train_dataloder = torch.utils.data.DataLoader(train_set,
                                                  batch_size=kwargs["extra_args"]["batch_size"],
                                                  shuffle=True)
    val_dataloder = torch.utils.data.DataLoader(val_set,
                                                batch_size=kwargs["extra_args"]["batch_size"],
                                                shuffle=True)
    # Choose and create the appropriate model based on the input arguments
    if kwargs["extra_args"]["model"] == 'cnn':
        model = CNNModel(kwargs["extra_args"]["cls_nums"]).to(device)
        opt.model_dir = os.path.join(opt.model_dir, 'cnn')
    elif kwargs["extra_args"]["model"] == 'resnet':
        model = ResnetModel(kwargs["extra_args"]["cls_nums"]).to(device)
        opt.model_dir = os.path.join(opt.model_dir, 'resnet')
    elif kwargs["extra_args"]["model"] == 'resnet50':
        model = models.resnet50(pretrained=True)
        opt.model_dir = os.path.join(opt.model_dir, 'resnet50')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, kwargs["extra_args"]["cls_nums"])
        model = model.to(device)

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=kwargs["extra_args"]["lr"],
                          momentum=kwargs["extra_args"]["momentum"],
                          weight_decay=kwargs["extra_args"]["weight_decay"])
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                          step_size=kwargs["extra_args"]["lr_decay_step"],
    #                                          gamma=kwargs["extra_args"]["lr_decay_factor"])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["extra_args"]["num_epochs"])
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_acc = 0
    best_model = model.state_dict()
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    # Train the model
    for epoch in range(kwargs["extra_args"]["num_epochs"]):
        start_time = time.time()
        print(f"------ Epoch {epoch} -----")
        # first train
        model.train()
        train_loss = 0
        train_acc = 0
        for data in train_dataloder:
            images, labels = data
            preds = model(images.to(device))
            loss = criterion(preds.cpu(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, pred_cls = torch.max(preds, 1)
            train_acc += torch.sum(pred_cls.cpu() == labels).to(torch.float32)

        lr_scheduler.step()
        train_loss_list.append(train_loss / train_num)
        train_acc_list.append(train_acc / train_num)
        print(f"train_loss = {train_loss_list[-1]}, train_acc = {train_acc_list[-1]}")

        # then validate
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for data in val_dataloder:
                images, labels = data
                preds = model(images.to(device))
                loss = criterion(preds.cpu(), labels)
                val_loss += loss.item()

                _, pred_cls = torch.max(preds, 1)
                val_acc += torch.sum(pred_cls.cpu() == labels).to(torch.float32)
            # save best model
            if val_acc > best_acc:
                best_acc = val_acc
                # best_model = model.state_dict()
            val_loss_list.append(val_loss / val_num)
            val_acc_list.append(val_acc / val_num)
            print(f"val_loss = {val_loss_list[-1]}, val_acc = {val_acc_list[-1]}")
        print(f"Time cost = {time.time() - start_time}")
        # Early stopping check
        early_stop = early_stopping(val_loss_list[-1], model)
        if early_stop:
            print("Early stopping triggered")
            break
    # Save the best model from the early stopping object
    save_model(early_stopping.best_model, opt.model)
    # save_model(model, opt.model)

    # Plot the training and validation loss and accuracy
    plt.figure(0)
    plt.plot(train_loss_list, 'r', label='train')
    plt.plot(val_loss_list, 'b', label='val')
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.figure(1)
    plt.plot(train_acc_list, 'r', label='train')
    plt.plot(val_acc_list, 'b', label='val')
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.legend()

    plt.show()
    return train_acc_list[-1].item()

def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    # Device configuration
    use_gpu = kwargs["extra_args"]["use_gpu"]
    if use_gpu:
        use_gpu = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if use_gpu else "cpu")
    print(f"device = {device}")

    # Load the pre-trained model
    model_path = os.path.join('models', kwargs["extra_args"]["model"], 'model.pth')
    if kwargs["extra_args"]["model"] == "cnn":
        model = CNNModel(kwargs["extra_args"]["cls_nums"])
        model = torch.load(model_path)
    elif kwargs["extra_args"]["model"] == "resnet":
        model = ResnetModel(kwargs["extra_args"]["cls_nums"])
        model = torch.load(model_path)
    elif kwargs["extra_args"]["model"] == "resnet50":
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, kwargs["extra_args"]["cls_nums"])
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Load the test dataset
    test_images, test_labels, _, _ = split_train_val_data(opt.test_data_dir, 1)

    test_num = len(test_labels)
    print(f"test_num = {test_num}")
    test_set = dataset(test_images, test_labels, 'val',
                       kwargs["extra_args"]["use_transforms"], kwargs["extra_args"]["input_size"])
    test_dataloader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=kwargs["extra_args"]["batch_size"],
                                                  shuffle=True)
    # Test the model
    start_time = time.time()
    model.eval()
    test_loss = 0
    test_acc = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            preds = model(images.to(device))
            loss = criterion(preds.cpu(), labels)
            test_loss += loss.item()

            _, pred_cls = torch.max(preds, 1)
            test_acc += torch.sum(pred_cls.cpu() == labels).to(torch.float32)

    test_loss /= len(test_set)
    test_acc /= len(test_set)
    print(f"Time cost = {time.time() - start_time}")
    print(f"test_loss = {test_loss}, test_acc = {test_acc}")
    print("===================================================")
    return test_acc.item()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train', 'test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pth', help='the pre-trained model')
    # ------zmy------
    # params for dataset
    parser.add_argument('--train_val_split', default=0.8, help='percent of train data')
    parser.add_argument('--use_transforms', default=True, help='use image transforms')
    # params for model
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'resnet50'],
                        help='choose the model architecture')
    parser.add_argument('--input_size', default=(224, 224), help='input size of model')
    parser.add_argument('--cls_nums', default=15, help='num of classes')
    # params for train
    parser.add_argument('--use_gpu', default=True, help='batch_size')
    parser.add_argument('--batch_size', default=32, help='batch_size')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', default=5e-4, help='L2 regularization')
    parser.add_argument('--lr_decay_step', default=40, help='lr_decay_step')
    parser.add_argument('--lr_decay_factor', default=0.8, help='lr_decay_factor')
    parser.add_argument('--num_epochs', default=200, help='epochs')
    # ---------------
    opt = parser.parse_args()

    extra_args = {"use_transforms": opt.use_transforms, "batch_size": opt.batch_size,
                  "use_gpu": opt.use_gpu, "input_size": opt.input_size, "cls_nums": opt.cls_nums,
                  "lr": opt.lr, "momentum": opt.momentum, "lr_decay_step": opt.lr_decay_step,
                  "lr_decay_factor": opt.lr_decay_factor, "num_epochs": opt.num_epochs,
                  "model_dir": opt.model_dir, "weight_decay": opt.weight_decay, "model": opt.model}
    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir, extra_args=extra_args)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir, extra_args=extra_args)
        print(testing_accuracy)
