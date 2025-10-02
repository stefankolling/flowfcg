"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
import requests
import zipfile
import pathlib

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

NUM_WORKERS = os.cpu_count()
DATA_PATH = "data"
DATASET_NAME = "fcg-ff-vf-vv"
EXTERNAL_DATASET_URL = "https://github.com/stefankolling/flowfcg/raw/refs/heads/main/data/fcg-ff-vf-vv.zip" # this code was written before we decided to clone th whole repository into the local workspace; now we could simply unzip the dataset already present locally

def load_data(dataset_name: str=DATASET_NAME,
              external_source: str=EXTERNAL_DATASET_URL,
              data_path=DATA_PATH
              ):
  """Loads a dataset consisting of train and test data from an external sourc

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    dataset_name: String literal to be used as name of dataset to be loaded
    external_source: URL to an external source with dataset as zip file (e.g. GitHub link)
    data_path: String literal to be used as local path for the dataset

  Returns:
    dataset_path: pathlib.Path object to dataset path
  """

  # Setup path to data folder
  data_path = Path(data_path + "/")
  dataset_path = data_path / dataset_name

  # If the dataset folder doesn't exist, download it and prepare it...
  if dataset_path.is_dir():
      print(f"{dataset_path} directory already exists.")
  else:
      print(f"Did not find {dataset_path} directory, creating one...", end="")
      dataset_path.mkdir(parents=True, exist_ok=True)
      print("done")

  # Download data
  with open(data_path / (dataset_name + ".zip"), "wb") as f:
      request = requests.get(external_source)
      print("Downloading dataset from external source...", end="")
      f.write(request.content)
      print("done")

  # Unzip external dataset
  with zipfile.ZipFile(data_path / (dataset_name + ".zip"), "r") as zip_ref:
      print("Unzipping dataset...", end="")
      zip_ref.extractall(dataset_path)
      print("done")

  # Cleaning up by removing zip file
  print("Cleaning up temporary zip file...", end="")
  os.remove(data_path / (dataset_name + ".zip"))
  print("done")
  print(f"Dataset {dataset_name} can now be found at '{str(dataset_path)}.'")

  return dataset_path

class MyDataset(Dataset):
  """MyDataset extends torch.utils.data.Dataset to allow applying transforms to a subset after the dataset
     has already been created. For example, this is useful if you first want to create a complete dataset without applying transforms,
     then split it into train and test (sub-)datasets and eventually apply different transforms to the subset
  """ 
  def __init__(self, subset, transform=None):
      self.subset = subset
      self.transform = transform
      
  def __getitem__(self, index):
      x, y = self.subset[index]
      if self.transform:  # If a transform is set, it will be applied to the element before returning it
          x = self.transform(x)
      return x, y
      
  def __len__(self):
      return len(self.subset)

def train_test_split_create_dataloaders(dataset_path: str,
                     train_ratio: float,
                     train_transform: transforms.Compose,
                     test_transform: transforms.Compose,
                     batch_size: int,
                     num_workers: int=NUM_WORKERS):
  """Splits dataset into train and test set according to given ratio

  Takes in a dataset path and a train or test ratio and splits the dataset into train and test set.

  Args:
    dataset_path: pathlib.Path object
    train_ratio: float in the range of 0.0 to 1.0
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = train_test_split_create_dataloaders(dataset_path=path/to/dir,
                             train_transform=some_transform,
                             test_transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  # Use ImageFolder to create the dataset
  dataset = datasets.ImageFolder(dataset_path)

  # Get class names
  class_names = train_data.classes

  # Split dataset
  assert train_ratio > 0.0 and train_ratio < 1.0, print(f"The train ratio must be in the range from 0.0 to 1.0, but was {train_ratio} instead.")
  dataset_size = len(dataset)
  train_size = int(train_ratio * dataset_size)
  test_size = dataset_size - train_size
  train_data, test_data = random_split(dataset, [train_size, test_size])

  # Apply different transforms to each subdataset
  train_data, test_data = MyDataset(train_data, transform=train_transform), MyDataset(test_data, transform=test_transform)

  # Turn images into data loaders
  train_dataloader = DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=False, # No need to shuffle test data
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
  
def create_dataloaders_from_directory(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    train_transform: torchvision transforms to perform on testing data.
    test_transform: torchvision transforms to perform on testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             train_transform=some_transform,
                             test_transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=train_transform)
  test_data = datasets.ImageFolder(test_dir, transform=test_transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names

