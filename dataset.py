import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class dataset(Dataset):
    def __init__(self, images, labels, mode='train', use_transforms=True, input_size=(224, 224)):
        '''
        Args:
            data_dir: directory of test/train dataset
            mode: 'train' or 'val'
        '''
        print("Build " + mode + " dataset...")
        super(dataset, self).__init__()
        self.images = images
        self.labels = labels
        self.mode = mode
        # basic transform
        self.Transform = {
            'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(input_size),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(input_size),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(input_size),
                transforms.ToTensor()
            ])
        }
        if use_transforms:
            self.Transform = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    # transforms.Resize(input_size),
                    transforms.Resize((int(input_size[0] * 1.2), int(input_size[1] * 1.2))),
                    transforms.RandomCrop(input_size),
                    transforms.ToTensor(),
                    # transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(30),
                    # transforms.RandomAffine(10),
                    # transforms.RandomPerspective(distortion_scale=0.2),
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ])
            }
        print("Finished")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_name)
        img = self.Transform[self.mode](img)
        return img, label
