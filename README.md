# ViT Fine-Tuner

## Overview

ViT Fine-Tuner is designed to simplify the fine-tuning of Vision Transformer (ViT) models. Leveraging the `transformers` library by Hugging Face and the `datasets` library, ViT Fine-Tuner streamlines the process of preparing datasets, setting up training arguments, and executing the fine-tuning of ViT models.

## Features

- **Automatic Dataset Splitting**: Automatically splits your training dataset into training and testing sets.
- **Pre-trained Model Loading**: Easily load pre-trained ViT models from Hugging Face.
- **Easy Training**: A VIT fine tuner that just works, i will expand this project to allow control over all the parameters, but the idea is that you can just drop the images that you want to use for fine tuning and run the main.py to have a usable model
- **Metrics and Visualization**: Compute accuracy metrics and visualize training progress at the end of the process.


## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/vit-finetuner.git
cd vit-finetuner
pip install -r requirements.txt
```


### Requirements
- torch
- numpy
- matplotlib
- evaluate
- datasets
- transformers


## Usage
Ensure your dataset is organized in the following structure:


```
dataset/
│
└───train/
│   └───class1/
│       │   img1.jpg
│       │   img2.jpg
│       │   ...
│   └───class2/
│       │   img1.jpg
│       │   img2.jpg
│       │   ...
│   ...

```
The load_dataset method will automatically create a test directory and move 20% of the training data into it if it doesn't already exist.

run:
```
python main.py
```