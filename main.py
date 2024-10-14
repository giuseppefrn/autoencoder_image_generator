import argparse
import os

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.upBlock_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # self.add_1 = nn.Add()
        # self.add_2 = nn.Add()
        # self.add_3 = nn.Add()

        self.last_conv = nn.Conv2d(8, 3, 3, 1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def main(self, input):
        x1 = self.block_1(input)
        x2 = self.block_2(x1)
        x = self.block_3(x2)

        x = self.upBlock_1(x)
        x = torch.add(x, x2)

        x = self.upBlock_2(x)
        x = torch.add(x, x1)

        x = self.upBlock_3(x)

        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x

    def forward(self, input):
        return self.main(input)


# Argument parser to accept paths from command-line
def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Image Generation Script")

    parser.add_argument('--model-path', type=str, default='model-best.pt', help='Path to the trained model (.pt file)')
    parser.add_argument('--input-image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-folder', type=str, default='outputs',
                        help='Folder where the output image will be saved')
    parser.add_argument('--output-name', type=str, default='output_image.png', help='Name for the output image file')

    return parser.parse_args()


# Preprocessing function for the input image
def preprocess_image(image_path):
    # Define the necessary transforms (modify as needed for your model)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match your autoencoder input size
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # Load and convert the image to RGB
    image = Image.open(image_path).convert('RGB')

    # Apply the transforms
    image = transform(image)

    # Add a batch dimension (autoencoders usually expect a batch input)
    image = image.unsqueeze(0)

    return image


# Postprocessing function to convert the output tensor to an image
def postprocess_image(output_tensor):
    # Remove the batch dimension and convert back to a PIL image
    output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
    output_tensor = output_tensor.cpu().detach()  # Move to CPU and detach from computation graph

    # Convert the tensor back to a PIL image
    output_image = transforms.ToPILImage()(output_tensor)

    return output_image


# Function to rename state_dict keys by adjusting upBlock layers
def adjust_state_dict_keys(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        if 'upBlock' in key:
            # Identify upBlock layers and adjust the second number
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                second_number = int(parts[1]) - 1  # Decrease the second number by 1
                parts[1] = str(second_number)
                new_key = '.'.join(parts)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def swap_state_dict_weights(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'upBlock' in key and 'weight' in key:
            if value.ndim > 1:  # Ensure the tensor has at least two dimensions
                new_state_dict[key] = value.transpose(0, 1)
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Main function for loading model, processing image, and saving output
def main():
    args = parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model (using the defined architecture)
    model = Generator(0)

    # Load the state_dict (weights) into the model
    try:
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))

        model.load_state_dict(state_dict)
        print(f'Successfully loaded model from {args.model_path}')
    except Exception as e:
        print(f"Failed to load model from {args.model_path}: {e}")
        return

    model.eval()  # Set the model to evaluation mode

    # Preprocess the input image
    try:
        input_image = preprocess_image(args.input_image)
    except Exception as e:
        print(f"Failed to preprocess image {args.input_image}: {e}")
        return

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    output_image_path = os.path.join(args.output_folder, args.output_name)

    # Move the model and input to the correct device (CPU or GPU)
    model = model.to(device)
    input_image = input_image.to(device)

    # Perform inference (image generation)
    try:
        with torch.no_grad():  # Disable gradient calculation for inference
            generated_image = model(input_image)

        # Post-process the generated image to convert it into a format that can be saved
        output_image = postprocess_image(generated_image)

        # Save the generated image to the output folder
        output_image.save(output_image_path)

        print(f'Generated image saved at: {output_image_path}')
    except Exception as e:
        print(f"Failed to generate or save the image: {e}")


# Run the main function if this script is executed
if __name__ == '__main__':
    main()
