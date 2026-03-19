import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''model and functions'''
activation = nn.ReLU()


class CAE(nn.Module):
    def __init__(self, cae_depth=64, cae_input_dimension=3, cae_strides=[3, 2, 2], cae_kernel_size=[5, 3, 3]):
        super(CAE, self).__init__()
        self.depth = cae_depth
        self.input_dimension = cae_input_dimension
        self.n_layers = len(cae_strides)
        self.layers = [int(2**i_layer * self.depth) for i_layer in range(self.n_layers)]
        self.strides = cae_strides
        if isinstance(cae_kernel_size, int):
            self.kernel_size = [cae_kernel_size] * self.n_layers
        else:
            self.kernel_size = cae_kernel_size
        
        self.encoder = nn.ModuleList()
        in_channels = self.input_dimension
        for i_layer in range(self.n_layers):
            self.encoder.append(
                nn.Conv3d(in_channels, 
                          self.layers[i_layer], 
                          kernel_size=self.kernel_size[i_layer], 
                          stride=self.strides[i_layer], 
                          padding=(self.kernel_size[i_layer]-1)//2)
            )
            in_channels = self.layers[i_layer]
        
        self.decoder = nn.ModuleList()
        for i_layer in range(self.n_layers-1, -1, -1):
            self.decoder.append(
                nn.ConvTranspose3d(self.layers[i_layer], 
                                   self.layers[i_layer-1] if i_layer > 0 else self.input_dimension, 
                                   kernel_size=self.kernel_size[i_layer], 
                                   stride=self.strides[i_layer], 
                                   padding=(self.kernel_size[i_layer]-1)//2, 
                                   output_padding=self.strides[i_layer]-1)
            )
        
    def forward(self, x):
        z = x.clone()
        # print(f'Input layer shape: {z.shape}')

        feature_layers = []
        # feature_layers.append(z)
        for i_layer, layer in enumerate(self.encoder):
            z = layer(z)
            feature_layers.append(z)
            # print(f'Encoder layer {i_layer} output shape: {z.shape}')
            z = activation(z)
        
        x_ = z.clone()
        for i_layer, layer in enumerate(self.decoder):
            x_ = layer(x_)
            if i_layer < len(self.decoder) - 1:
                x_ = activation(x_)
            # print(f'Decoder layer {i_layer} output shape: {x_.shape}')
        
        return feature_layers, z, x_
