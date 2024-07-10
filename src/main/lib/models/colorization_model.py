from src.main.lib.utils.imports import *

class CNN(nn.Module):
    def __init__(self, batch_size : int,
                  num_layers : int,
                  inp_size : tuple[int],
                  leaky_relu : bool,
                  batch_norm : bool,
                  device : torch.DeviceObjType,
                  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Model Size Parameters
        self.B = batch_size # Batch Size B
        self.L = num_layers # Conv Layers L
        self.I = inp_size # Input Size I

        # Model architecture parameters
        self.leaky_relu = leaky_relu
        self.batch_norm = batch_norm

        # Device
        self.device = device

        # Defining the CNN
        self.model_blocks = nn.ModuleList([])

        in_channels = self.I[0] # Should be one for grayscale input
        out_channels = 16

        for _ in range(self.L):
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding="same")
            )

            if self.leaky_relu:
                block.add_module(
                    "leaky_relu",
                    nn.LeakyReLU()
                )
            
            if self.batch_norm:
                block.add_module(
                    "batch_norm",
                    nn.BatchNorm2d(num_features=out_channels)
                )
            
            self.model_blocks.append(block)
            in_channels = out_channels
            out_channels *= 2
 
    def forward(self, x):
        for block in self.model_blocks:
            x = block(x)
        
        return nn.Tanh()(x)