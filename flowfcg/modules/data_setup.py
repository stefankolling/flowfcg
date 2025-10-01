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
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
DATA_PATH = "data"
DATASET_NAME = "fcg-ff-vf-vv"
EXTERNAL_DATASET_URL = "https://github.com/stefankolling/flowfcg/raw/refs/heads/main/data/fcg-ff-vf-vv.zip"

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

def train_test_split(dataset_path: pathlib.Path,
                       train_ratio: float,
                       test_ratio: float):
  """Splits dataset into train and test set according to given ratio

  Takes in a dataset path and a train or test ratio and splits the dataset into train and test set.

  Args:
    dataset_path: pathlib.Path object
    train_ratio: float in the range of 0.0 to 1.0
    test_ratio: float in the range of 0.0 to 1.0

  Returns:
    train_dir: pathlib.Path object to train dir
    test_dir: pathlib.Path object to test dir
  """

  # ToDo: Implement function
  train_dir = Path("")
  test_dir = Path("")

  return train_dir, test_dir

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform) # currently the same transform is performed ob train and test data

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
