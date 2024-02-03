import torch
import torch.nn as nn
import torch.nn.functional as F

from Evaluation import net_evaluation


class Caries_SegUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Caries_SegUNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2


def Model_Caries_SegNet(Images, Mask, height, width):
    # Example usage:
    # Define the model
    in_channels = 3  # Number of input channels
    out_channels = 1  # Number of output channels for binary segmentation

    model = Caries_SegUNet(in_channels, out_channels)

    # Example input tensor
    batch_size = 1
    input_tensor = torch.rand((batch_size, in_channels, height, width))

    # Forward pass
    output = model(input_tensor)

    image_count = Images.shape[0]
    results = model.predict_generator(output, Mask, verbose=1)
    Eval = net_evaluation(Images, Mask)
    return results, Eval
