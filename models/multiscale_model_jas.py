# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:32:24 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
# band number is 224, endmember mumber is 4
import torch
from torch.nn import Module, Sequential, Conv3d, ReLU, LeakyReLU, Linear, Sigmoid, Conv2d, Softmax, BatchNorm3d


# Define the network
class autoencoder_model(Module):
    def __init__(self):
        super(autoencoder_model, self).__init__()

        self.encoder_cnn = Sequential(
            Conv3d(
                in_channels=1, out_channels=128, kernel_size=(3, 3, 6), stride=(1, 1, 2), padding=(1, 1, 0), bias=False
            ),
            ReLU(),  # layer1 111
            Conv3d(
                in_channels=128, out_channels=64, kernel_size=(3, 3, 4), stride=(1, 1, 2), padding=(1, 1, 0), bias=False
            ),
            ReLU(),  # layer2 55
            Conv3d(
                in_channels=64, out_channels=32, kernel_size=(3, 3, 5), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),  # layer3 27
            Conv3d(
                in_channels=32, out_channels=16, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),  # layer4  13
            Conv3d(
                in_channels=16, out_channels=8, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),  # layer5  6
            Conv3d(
                in_channels=8, out_channels=3, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=False
            )
        )

        self.decoder_linear = Sequential(
            Linear(3, 156*3, bias=False),
            #   ReLU(True)
        )
        self.decoder_nonlinear = Sequential(
            Linear(156*3, 156, bias=True),
            Sigmoid(),
            Linear(156, 156, bias=True),
            Sigmoid(),
            Linear(156, 156, bias=True)
        )

    def forward(self, x):
        ######################################
        x = torch.reshape(x, (-1, 1, 3, 3, 156))
        out_encoder = self.encoder_cnn(x)
        out_encoder = torch.reshape(out_encoder, (-1, 3))
        out_encoder = out_encoder.abs()
        out_encoder = out_encoder.t() / out_encoder.sum(1)
        out_encoder = out_encoder.t()
        out_linear = self.decoder_linear(out_encoder)
        out_nonlinear = self.decoder_nonlinear(out_linear)

        return out_linear, out_nonlinear, out_encoder

    def get_endmember(self, x):
        endmember = self.decoder_linear(x)
        return endmember

    def get_abundance(self, x):
        x = self.encoder_cnn(x)
        x = torch.reshape(x, (-1, 4))
        weights = self.encoder_cnn(x)
        weights = torch.reshape(weights, (-1, 156))
        weights = weights.abs()
        weights = weights.t() / weights.sum(1)
        weights = weights.t()
        return weights
