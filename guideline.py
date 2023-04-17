"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Zheng Huan <huan_zheng@u.nus.edu>
"""
import os
import json
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import dataset
from model.Resnet import BasicBlockModel as Model
# from model.BasicCNN import Model
with open("classes.json") as f:
    classes = json.load(f)


torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


###################################### Subroutines #####################################################################
"""
Example of subroutines you might need. 
You could add/modify your subroutines in this section. You can also delete the unnecessary functions.
It is encouraging but not necessary to name your subroutines as these examples. 
"""
def build_vocabulary(**kwargs):
    pass

def get_hist(**kwargs):
    pass

def classifier(**kwargs):
    pass

def get_accuracy(**kwargs):
    pass

def save_model(model, model_dir):
    torch.save(model, model_dir)


def split_train_val_data(train_data_dir, train_val_split):
    print("Split data...")
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    for i in range(len(classes)):
        sub_class = classes[str(i + 1)]
        sub_dir = os.path.join(train_data_dir, sub_class)
        image_names = os.listdir(sub_dir)
        train_num = int(len(image_names) * train_val_split)
        random.shuffle(image_names)
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
    # device
    use_gpu = kwargs["extra_args"]["use_gpu"]
    if use_gpu:
        use_gpu = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if use_gpu else "cpu")
    print(f"device = {device}")
    # build dataset
    train_images, train_labels, val_images, val_labels = split_train_val_data(opt.train_data_dir, opt.train_val_split)
    train_num, val_num = len(train_labels), len(val_labels)
    print(f"train_num = {train_num}, val_num = {val_num}")
    train_set = dataset(train_images, train_labels, 'train',
                        kwargs["extra_args"]["use_transforms"], kwargs["extra_args"]["input_size"])
    val_set = dataset(val_images, val_labels, 'val',
                      kwargs["extra_args"]["use_transforms"], kwargs["extra_args"]["input_size"])
    # build data loader
    train_dataloder = torch.utils.data.DataLoader(train_set,
                                                  batch_size=kwargs["extra_args"]["batch_size"],
                                                  shuffle=True)
    val_dataloder = torch.utils.data.DataLoader(val_set,
                                                batch_size=kwargs["extra_args"]["batch_size"],
                                                shuffle=True)
    # model
    # model = Model(kwargs["extra_args"]["cls_nums"]).to(device)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, kwargs["extra_args"]["cls_nums"])
    model = model.to(device)

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
                best_model = model.state_dict()
            val_loss_list.append(val_loss / val_num)
            val_acc_list.append(val_acc / val_num)
            print(f"val_loss = {val_loss_list[-1]}, val_acc = {val_acc_list[-1]}")
        print(f"Time cost = {time.time() - start_time}")
    save_model(best_model, opt.model_dir)
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
    pass



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pth', help='the pre-trained model')
    # ------zmy------
    # params for dataset
    parser.add_argument('--train_val_split', default=0.8, help='percent of train data')
    parser.add_argument('--use_transforms', default=True, help='use image transforms')
    # params for model
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
    parser.add_argument('--num_epochs', default=500, help='epochs')
    # ---------------
    opt = parser.parse_args()

    if opt.phase == 'train':
        extra_args = {"use_transforms": opt.use_transforms, "batch_size": opt.batch_size,
                      "use_gpu": opt.use_gpu, "input_size": opt.input_size, "cls_nums": opt.cls_nums,
                      "lr": opt.lr, "momentum": opt.momentum, "lr_decay_step": opt.lr_decay_step,
                      "lr_decay_factor": opt.lr_decay_factor, "num_epochs": opt.num_epochs,
                      "model_dir": opt.model_dir, "weight_decay": opt.weight_decay}
        training_accuracy = train(opt.train_data_dir, opt.model_dir, extra_args=extra_args)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)






