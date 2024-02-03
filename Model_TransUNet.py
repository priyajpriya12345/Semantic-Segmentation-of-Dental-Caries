import torch
from monai.networks.nets import TransUNet
from Evaluation import net_evaluation


def Model_TransUNet(Images, Mask, height, width):
    # Create a TransUNet model with default settings
    model = TransUNet()

    # Example input tensor
    batch_size, channels = 1, 1
    input_tensor = torch.rand((batch_size, channels, height, width))

    # Forward pass
    output = model(input_tensor)
    image_count = Images.shape[0]
    results = model.predict_generator(output, Mask, verbose=1)
    Eval = net_evaluation(Images, Mask)
    return results


