# Autoencoder Image Generation Script

This repository contains a PyTorch-based implementation of an autoencoder model for image generation. The script allows you to load a trained model, process an input image, and generate an output image. This is handled via command-line arguments for specifying paths and filenames.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)

## Requirements

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Before running the code, ensure that you have Poetry installed on your system.
After installing the Poetry package, you can install the required dependencies by running the following command:

```poetry install```

Ensure you are using the correct Python version (3.8) by running.


## Usage
The script will load the model from the specified path, process the input image, and save the generated image to the output folder with the specified name.

Example:

```python main.py --model-path path_to_your_model.pt --input-image path_to_input_image --output-folder path_to_output_folder --output-name output_image.png```

Arguments:
- `--model-path`: Path to the trained model file.
- `--input-image`: Path to the input image file.
- `--output-folder`: Path to the output folder where the generated image will be saved.
- `--output-name`: Name of the generated image file.
