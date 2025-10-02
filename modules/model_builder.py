"""
Contains PyTorch model code to instantiate various CNN models
"""
import torch
from torch import nn
import torchvision
from torchvision import transforms

RESIZE_DIM = (64,64)

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__() # immediately call __init__() of super class
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.LazyLinear(out_features=output_shape) # nn.LazyLinear() automatically infers in_features from input shape of the tensor coming from the previous layer; other than that identical to nn.Linear()
      )

  def forward(self, x: torch.Tensor):
      return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

  def transforms(self):
      # Create manual transforms
      manual_transforms = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.ToTensor()  # this also automatically normalises RGB values from range 0..255 to 0..1
      ])
      return manual_transforms

class FcgCnn(nn.Module):
  """Creates the FCG-CNN architecture.

  Replicates the FCG-CNN architecture from the Udemy course by Kyrill Emerenko.
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__() # immediately call __init__() of super class
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )

      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.LazyLinear(out_features=128),
          nn.ReLU(),
          nn.Linear(in_features=128,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      return self.classifier(self.conv_block_1(x)) # <- leverage the benefits of operator fusion

  def transforms(self):
      # Create manual transforms
      manual_transforms = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.ToTensor()  # this also automatically normalises RGB values from range 0..255 to 0..1
      ])

      return manual_transforms

class PretrainedCNN():
  weights = None
  model = None

  def __init__(self, model: str, output_shape: int):

    match model:
      case "vgg16":
        self.weights =  torchvision.models.VGG16_Weights.DEFAULT #torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
        self.model = torchvision.models.vgg16(weights=self.weights)
      case "efficientnet_b0":
        self.weights =  torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
        self.model = torchvision.models.efficientnet_b0(weights=self.weights)
      case "mobilenet_v2":
        self.weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        self.model = torchvision.models.mobilenet_v2(weights=self.weights)
      case _:
        print("[Info] No corresponding model found.")

    # Freezing all layers (so that these parameters will not be trained anymore)
    for param in self.model.features.parameters():
      param.requires_grad = False

    # Recreate the same classifier layer with 3 output classes (this will be trainable) for all models above and set it to the target device
    self.model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.LazyLinear(out_features=output_shape, # same number of output units as our number of classes
                        bias=True))

  def transforms(self):
	  return self.weights.transforms()
