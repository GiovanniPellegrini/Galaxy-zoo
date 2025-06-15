from torch import nn, Tensor
import torch
from typing import Literal
from dataclasses import dataclass
from galaxy_classification.training_utils import EPS, GAMMA


"""
Class to initialize the configuration of the CNN.
It consists of the following parameters:
- network_id: The id of the network.
- channel_count_hidden: The number of channels in the hidden layers.
- convolution_kernel_size: The size of the convolution kernel.
- mlp_hidden_unit_count: The number of hidden units in the MLP.
"""
@dataclass
class GalaxyClassificationCNNConfig:
    network_id: Literal["classification"]
    channel_count_hidden: int
    convolution_kernel_size: int
    mlp_hidden_unit_count: int

@dataclass
class GalaxyRegressionCNNConfig:
    network_id: Literal["regression"]
    channel_count_hidden: int
    convolution_kernel_size: int
    mlp_hidden_unit_count: int

"""
Class to initialize the double convolution block. 
It consists of two convolutional layers with batch normalization and ReLU activation.
"""
class DoubleConvolutionBlock(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        channel_count_hidden: int,
        kernel_size: int,
    ):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_count_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(channel_count_hidden)
        self.conv2=nn.Conv2d(
            in_channels=channel_count_hidden,
            out_channels=channel_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(channel_out)
        
        if channel_in != channel_out:
            self.res_conv = nn.Conv2d(channel_in, channel_out, kernel_size=1)
        else:
            self.res_conv = None
        self.drop = nn.Dropout2d(0.1)   

    def forward(self, image: Tensor):
        image_convolved = nn.functional.relu(self.bn1(self.conv1(image)))
        image_convolved = nn.functional.relu(self.bn2(self.conv2(image_convolved)))

        residual = image if self.res_conv is None else self.res_conv(image)
        return image_convolved + residual
    

"""
Model class for the galaxy classification CNN.
It consists of the following layers:
- DoubleConvolutionBlock: Two convolutional layers with batch normalization and ReLU activation.
- AvgPool2d: Average pooling layer.
- DoubleConvolutionBlock
- AvgPool2d
- Flatten: Flatten the output of the last convolutional layer.
- Linear: Fully connected layer with ReLU activation and dropout.
- Linear: Fully connected layer with softmax activation: 3 output channels.
"""
class GalaxyClassificationCNN(nn.Module):
    def __init__(
        self,
        image_input_shape: tuple[int, int],
        channel_count_hidden: int,
        convolution_kernel_size: int,
        mlp_hidden_unit_count: int,
    ):
        super().__init__()
        POOLING_COUNT = 2
        GALAXY_CHANNEL_COUNT=2

        mlp_feature_count = (
            image_input_shape[0] // (2 ** POOLING_COUNT)* (image_input_shape[1] // (2 ** POOLING_COUNT)) * channel_count_hidden
        )
        self.feature = nn.Sequential(
            DoubleConvolutionBlock(
                channel_in=3,
                channel_out=channel_count_hidden,
                channel_count_hidden=channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            nn.AvgPool2d(kernel_size=(2,2)),
            DoubleConvolutionBlock(
                channel_in=channel_count_hidden,
                channel_out=channel_count_hidden,
                channel_count_hidden=channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            nn.AvgPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(mlp_feature_count, mlp_hidden_unit_count),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_unit_count, GALAXY_CHANNEL_COUNT)
        )
        

        
    def forward(self, image: Tensor):
        image = self.feature(image)
        return image
    
"""
Model class for the galaxy regression CNN.
It consists of a backbone:
- DoubleConvolutionBlock: Two convolutional layers with batch normalization and ReLU activation.
- AvgPool2d: Average pooling layer.
- DoubleConvolutionBlock
- AvgPool2d
- Flatten: Flatten the output of the last convolutional layer.
- Linear: Fully connected layer with ReLU activation and dropout.
- Linear: Fully connected layer with softmax activation: 3 output channels.
And different multiple heads for regression:
- head_q1: Linear layer with 3 output channels and sigmoid activation.
- head_q2: Linear layer with 2 output channels and sigmoid activation.
- head_q7: Linear layer with 3 output channels and sigmoid activation.
"""        


class GalaxyRegressionCNN(nn.Module):
    def __init__(
        self,
        image_input_shape: tuple[int, int],
        channel_count_hidden: int,
        convolution_kernel_size: int,
        mlp_hidden_unit_count: int,
    ):
        super().__init__()
        POOLING_COUNT = 2

        mlp_feature_count = (
            image_input_shape[0] // (2 ** POOLING_COUNT)
            * image_input_shape[1] // (2 ** POOLING_COUNT)
            * channel_count_hidden
        )

        self.backbone = nn.Sequential(
            DoubleConvolutionBlock(
                channel_in=3,
                channel_out=channel_count_hidden,
                channel_count_hidden=int(channel_count_hidden*1.5),
                kernel_size=convolution_kernel_size,
            ),
            nn.AvgPool2d(kernel_size=(2, 2)),
            DoubleConvolutionBlock(
                channel_in=channel_count_hidden,
                channel_out=channel_count_hidden,
                channel_count_hidden=channel_count_hidden*2,
                kernel_size=convolution_kernel_size,
            ),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(mlp_feature_count, mlp_hidden_unit_count),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Regression heads (sigmoid per output in [0,1])
        self.head_q1 = nn.Sequential(
            nn.Linear(mlp_hidden_unit_count, int(mlp_hidden_unit_count/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_unit_count/2), 3),
            nn.Softmax(dim=1)
        )
        self.head_q2 = nn.Sequential(
            nn.Linear(mlp_hidden_unit_count, int(mlp_hidden_unit_count/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_unit_count/2), 2),
            nn.Softmax(dim=1)
        )
        self.head_q6 = nn.Sequential(
            nn.Linear(mlp_hidden_unit_count, int(mlp_hidden_unit_count/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_unit_count/2), 2),
            nn.Softmax(dim=1)
        )
        self.head_q7 = nn.Sequential(
            nn.Linear(mlp_hidden_unit_count, int(mlp_hidden_unit_count/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_unit_count/2), 3),
            nn.Softmax(dim=1)
        )
        self.head_q8 = nn.Sequential(
            nn.Linear(mlp_hidden_unit_count, int(mlp_hidden_unit_count/2)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_unit_count/2), 7),
            nn.Softmax(dim=1)
        )

    def forward(self, image: Tensor):
        features = self.backbone(image)
        q1 = self.head_q1(features)
        q2 = self.head_q2(features) * q1[:, 1].unsqueeze(1) 
        q7 = self.head_q7(features) * q1[:, 0].unsqueeze(1)  
        q6 = self.head_q6(features) * (1 - q1[:, 2]).unsqueeze(1)  
        q8 = self.head_q8(features) * q6[:, 1].unsqueeze(1)
        
        def power_on_cols(t: torch.Tensor, idxs: list[int]) -> torch.Tensor:
            t = t.clone()
            t[:, idxs] = torch.clamp(t[:, idxs], min=EPS).pow(GAMMA)
            return t

        q1 = power_on_cols(q1, [2])
        q2 = power_on_cols(q2, [1])
        q7 = power_on_cols(q7, [0, 1])
        q8= power_on_cols(q8, [0, 1, 2, 3, 4, 5, 6])
        

        

        return {
            "q1": q1,
            "q2": q2,
            "q6": q6,
            "q7": q7,
            "q8": q8
        }


