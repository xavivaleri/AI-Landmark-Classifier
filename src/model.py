import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
    
        self.model = nn.Sequential(
         
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
  
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            
            nn.Flatten(),
            nn.Dropout(p=dropout),
            
            nn.Linear(7 * 7 * 256, out_features=500),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(500),
            nn.Dropout(p=dropout),
            
            nn.Linear(500, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            
            nn.Linear(256, num_classes)
        )
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
  
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
