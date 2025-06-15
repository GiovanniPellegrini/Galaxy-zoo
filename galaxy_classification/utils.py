"utils for galaxyzoo project"
import torch as T
import glob
import pandas as pd

# Create PyTorch dataset from list of files, apply image loading and decoding
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")  # Load and decode image
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, T.tensor(label)

def img_label(img,labels_df):
    '''Function to turn image path into ID, and return labels from labels_df'''
    id = int(img.split('/')[-1].split('.')[0])
    return labels_df.loc[id]

def trim_file_list(files,labels_df):
    '''Function to trim a list of files based on whether the file name is in the ID of labels_df'''
    files = [file for file in files if int(file.split('/')[-1].split('.')[0]) in labels_df.index]
    return files



