import torch
import segmentation_models_pytorch as smp
from Evaluation import net_evaluation


def Model_DANet(Images, Mask, height, width):
    # Define the model
    model = smp.DANet(encoder_name="resnet50", in_channels=3, classes=21)  # Adjust in_channels and classes as needed

    # Example input tensor
    batch_size = 1
    input_tensor = torch.rand((batch_size, 3, height, width))

    # Forward pass
    output = model(input_tensor)
    image_count = Images.shape[0]
    results = model.predict_generator(output, image_count, verbose=1)
    Eval = net_evaluation(Images, Mask)
    return results, Eval
