
import numpy as np
import glob
import torch
from typing import Self
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from PIL import Image
from typing import Any
from galaxy_classification.utils import trim_file_list, img_label
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from galaxy_classification.training_utils import EPS, GAMMA, CLASS_GROUPS, CLASS_GROUPS_TRANSFORMED



# --- Label preprocessing constants and groups ---



"""
Class to load images into list of strings and labels into a dataframe.
The images are discarded if they are not in the labels dataframe.
"""
@dataclass
class GalaxyDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame

    @classmethod
    def load(cls,image_path:str,label_path:str) -> Self:
        """
        Load the galaxy dataset from the given path.
        """
        file_paths = glob.glob(f"{image_path}/*.jpg")
        labels_df = pd.read_csv(label_path).set_index("GalaxyID")
        file_paths = trim_file_list(file_paths, labels_df)
        return cls(images=file_paths, labels=labels_df)
    def __len__(self):
        return len(self.images)


#       ------------------------ CLASSIFICATION ------------------------
"""
Class to convert the images into tensors and apply the transformations.
The main transformations are:
- CenterCrop
- Resize
- ToTensor
The training transformations are:
- CenterCrop
- Resize
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- ToTensor
The images are converted to RGB.
"""
@dataclass
class PreparedGalaxyClassificationDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame
    transform: Any

    @classmethod
    def from_unprepared(cls, dataset: GalaxyDataset, train:bool=True) -> Self:
        if train:
            # Apply transformations for training
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #modify randomnly the brightness, contrast and saturation of the image
                transforms.ToTensor() #convert the pixel values to float between 0 and 1
            ])
        else:          
            transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((64, 64)),
            transforms.ToTensor()  #convert the pixel values to float between 0 and 1
        ])
        return cls(images=dataset.images, labels=dataset.labels, transform=transform) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        # 
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   

        
        id = int(img_path.split("/")[-1].split(".")[0])
        label = int(self.labels.loc[id]["label"])
        return dict(images=image, labels=torch.tensor(label, dtype=torch.long))
"""
Class to split the dataset into training, validation and test sets.
It uses a GalaxyDataset as input and PreparedGalaxyDataset to transform the images.
It returns three dataloaders for training, validation and test sets.
"""    
@dataclass
class SplitGalaxyClassificationDataSet:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, dataset:GalaxyDataset, batch_size: int=32, validation_fraction:float=0.1, test_fraction:float=0.1):
         # Compute split sizes
        n = len(dataset)
        val_size = int(validation_fraction * n)
        test_size = int(test_fraction * n)
        train_size = n - val_size - test_size

        # Generate a random permutation of indices
        indices = torch.randperm(n).tolist()
        train_idx = indices[:train_size]
        val_idx = indices[train_size: train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Slice images and labels
        train_imgs = [dataset.images[i] for i in train_idx]
        val_imgs = [dataset.images[i] for i in val_idx]
        test_imgs = [dataset.images[i] for i in test_idx]

        train_lbls = dataset.labels.iloc[train_idx]
        val_lbls = dataset.labels.iloc[val_idx]
        test_lbls = dataset.labels.iloc[test_idx]

        # Build pure GalaxyDataset subsets
        train_base = GalaxyDataset(images=train_imgs, labels=train_lbls)
        val_base = GalaxyDataset(images=val_imgs, labels=val_lbls)
        test_base = GalaxyDataset(images=test_imgs, labels=test_lbls)

        # Apply preparation transforms
        train_ds = PreparedGalaxyClassificationDataset.from_unprepared(train_base, train=True)
        val_ds = PreparedGalaxyClassificationDataset.from_unprepared(val_base, train=False)
        test_ds = PreparedGalaxyClassificationDataset.from_unprepared(test_base, train=False)


        # Create DataLoaders
        self.training_dataloader = DataLoader(train_ds,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              persistent_workers=False)
        self.validation_dataloader = DataLoader(val_ds,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                persistent_workers=False)
        self.test_dataloader = DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          persistent_workers=False)


#       ------------------------ REGRESSION ------------------------

"""
Class to convert the images into tensors and apply the transformations.
The main transformations are:
- CenterCrop
- Resize
- ToTensor
The training transformations are:
- CenterCrop
- Resize
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- ToTensor
The images are converted to RGB.

The labels are preprocessed by clipping the values to a minimum of EPS, raising them to the power of GAMMA, and normalizing them.
"""
@dataclass
class PreparedGalaxyRegressionDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame
    transform: Any

    @classmethod
    def from_unprepared(cls, dataset: GalaxyDataset, train:bool=True) -> Self:
        
        
        
        if train:
            # Apply transformations for training
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #modify randomnly the brightness, contrast and saturation of the image
                transforms.ToTensor() #convert the pixel values to float between 0 and 1
            ])
        else:          
            transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((64, 64)),
            transforms.ToTensor()  #convert the pixel values to float between 0 and 1
        ])
        labels = dataset.labels.copy()
        #
        for col in CLASS_GROUPS_TRANSFORMED.values():
            labels[col] = labels[col].clip(lower=EPS)
            labels[col] = labels[col].pow(GAMMA)    
            
        return cls(images=dataset.images, labels=labels, transform=transform) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   

        
        id = int(Path(img_path).stem)
        row = self.labels.loc[id]
        
        q1 = torch.tensor([row["Class1.1"], row["Class1.2"], row["Class1.3"]], dtype=torch.float32)
        q2 = torch.tensor([row["Class2.1"], row["Class2.2"]], dtype=torch.float32)
        q6 = torch.tensor([row["Class6.1"], row["Class6.2"]], dtype=torch.float32)
        q7 = torch.tensor([row["Class7.1"], row["Class7.2"], row["Class7.3"]], dtype=torch.float32)
        q8 = torch.tensor([row["Class8.1"], row["Class8.2"], row["Class8.3"], 
                           row["Class8.4"], row["Class8.5"], row["Class8.6"], 
                           row["Class8.7"]], dtype=torch.float32)

        return {
            "images": image,
            "labels": {
                "q1": q1,
                "q2": q2,
                "q6": q6,
                "q7": q7,
                "q8": q8
            }
        }
        
@dataclass
class SplitGalaxyRegressionDataset:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader
    loss_weights:dict[str, float]= field(init=False)

    def __init__(self, dataset: GalaxyDataset, batch_size: int = 256, validation_fraction: float = 0.1, test_fraction: float = 0.1):
         # Compute split sizes
        n = len(dataset)
        val_size = int(validation_fraction * n)
        test_size = int(test_fraction * n)
        train_size = n - val_size - test_size

        # Generate a random permutation of indices
        indices = torch.randperm(n).tolist()
        train_idx = indices[:train_size]
        val_idx = indices[train_size: train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Slice images and labels
        train_imgs = [dataset.images[i] for i in train_idx]
        val_imgs = [dataset.images[i] for i in val_idx]
        test_imgs = [dataset.images[i] for i in test_idx]

        train_lbls = dataset.labels.iloc[train_idx]
        val_lbls = dataset.labels.iloc[val_idx]
        test_lbls = dataset.labels.iloc[test_idx]

        # Build pure GalaxyDataset subsets
        train_base = GalaxyDataset(images=train_imgs, labels=train_lbls)
        val_base = GalaxyDataset(images=val_imgs, labels=val_lbls)
        test_base = GalaxyDataset(images=test_imgs, labels=test_lbls)

        # Apply preparation transforms
        train_ds = PreparedGalaxyRegressionDataset.from_unprepared(train_base, train=True)
        val_ds = PreparedGalaxyRegressionDataset.from_unprepared(val_base, train=False)
        test_ds = PreparedGalaxyRegressionDataset.from_unprepared(test_base, train=False)


        # Create DataLoaders
        self.training_dataloader = DataLoader(train_ds,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              persistent_workers=False)
        self.validation_dataloader = DataLoader(val_ds,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                persistent_workers=False)
        self.test_dataloader = DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          persistent_workers=False)

        # Compute loss weights based on training labels inverting the preprocessing
        #labels_df = train_ds.labels.copy()
        #labels_inv = labels_df.clip(lower=EPS).pow(1.0 / GAMMA)
        #means = labels_inv.mean(axis=0)
        #inv = 1.0 / (means + EPS)
        #inv = inv / inv.mean()
        #self.loss_weights = inv.to_dict()