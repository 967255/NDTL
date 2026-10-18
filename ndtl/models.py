import torch
import torch.nn as nn

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

        feature_layers = []
        # feature_layers.append(z)
        for i_layer, layer in enumerate(self.encoder):
            z = layer(z)
            feature_layers.append(z)

            z = activation(z)

        x_ = z.clone()
        for i_layer, layer in enumerate(self.decoder):
            x_ = layer(x_)
            if i_layer < len(self.decoder) - 1:
                x_ = activation(x_)

        return feature_layers, z, x_

class EcologyCAE(CAE):
    """HP parameter names over the shared convolutional architecture."""

    def __init__(self, depth=96, input_channels=3, strides=(2, 2, 2), kernels=(5, 3, 3)):
        if len(strides) != len(kernels):
            raise ValueError("strides and kernels must have the same length")
        super().__init__(cae_depth=depth, cae_input_dimension=input_channels,
                         cae_strides=strides, cae_kernel_size=kernels)
        self.input_channels = self.input_dimension
        self.channels = self.layers
        self.kernels = self.kernel_size
        self.activation = activation

def load_checkpoint(model, path, device="cpu"):
    """Load a plain state dict or a checkpoint with model/state_dict metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.to(device).eval()
    return checkpoint
