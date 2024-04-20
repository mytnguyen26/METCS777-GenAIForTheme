from datasets import load_dataset
import pyarrow.parquet as pq
import boto3
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from transformers import CLIPTokenizer
import os

s3_client = boto3.client('s3')

# Name of the S3 bucket where the files are stored
bucket_name = 'caozhen'

# The path in the S3 bucket where the raw data is located
raw_data_path = 'raw_data/harvard/paintings/'

# The path in the S3 bucket where the train set will be saved, including the file name
train_set_save_path = 'train_set/train_set.parquet'

local_folder = 'paintings'
os.makedirs(local_folder, exist_ok=True)


def download_s3_folder(bucket_name, s3_folder, local_dir):
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                file_key = file['Key']
                if not file_key.endswith('/'):
                    local_file_path = os.path.join(local_dir, os.path.relpath(file_key, s3_folder))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    s3_client.download_file(bucket_name, file_key, local_file_path)


download_s3_folder(bucket_name, raw_data_path, local_folder)

img_dataset = load_dataset("imagefolder", data_dir=local_folder, split="train")


def resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0):
    """
    Resize an image to make its longest side equal to the original image's longest side,
    pad the shorter side to make the image a square, then resize to (512, 512).

    Args:
        img (PIL.Image): The image to resize and pad.
        final_size (tuple): The desired output size (height, width).
        padding_mode (str): Type of padding. Options include 'constant', 'edge', etc.
        fill (int, tuple): Pixel fill value for constant padding.

    Returns:
        PIL.Image: The resized and padded, then resized image.
    """
    original_width, original_height = img.size
    max_side = max(original_width, original_height)

    # Determine new size keeping aspect ratio
    if original_width > original_height:
        scale = max_side / original_width
        new_width = max_side
        new_height = int(original_height * scale)
    else:
        scale = max_side / original_height
        new_height = max_side
        new_width = int(original_width * scale)

    # Resize the image to max_side to keep aspect ratio
    img = F.resize(img, (new_height, new_width), interpolation=T.InterpolationMode.LANCZOS)

    # Calculate padding amounts
    pad_width = (max_side - new_width) // 2
    pad_height = (max_side - new_height) // 2

    # Apply padding to make it a square
    img = F.pad(img, [pad_width, pad_height, pad_width, pad_height], padding_mode=padding_mode, fill=fill)

    # Final resize to the desired output size
    img = F.resize(img, final_size, interpolation=T.InterpolationMode.LANCZOS)
    return img


# Setup the transformation pipeline with the updated function
transform_pipeline = T.Compose([
    T.Lambda(lambda img: resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


def preprocess_train(examples):
    # examples are a batch of 4 images
    # we apply the transformation (reference above for what it transfomed to)
    # then apply the tokenization
    examples["pixel_values"] = [transform_pipeline(image) for image in examples["image"]]
    inputs = tokenizer([example for example in examples["caption"]],
                       padding="max_length",
                       truncation=True,
                       return_tensors="pt")

    examples["tokens"] = inputs.input_ids
    return examples


img_dataset_transform_1 = img_dataset.map(preprocess_train, remove_columns=["image", "caption"], batched=True)


def preprocess_train_again(examples):
    # transform the image to tensor
    examples["image_tensor"] = [torch.tensor(image) for image in examples["pixel_values"]]
    return examples


train_set_2 = img_dataset_transform_1.map(preprocess_train_again, remove_columns=["pixel_values"], batched=True)

# Upload to S3 parquet
pq.write_table(train_set_2.data.table, 'train_set.parquet')

s3_client.upload_file('train_set.parquet', bucket_name, train_set_save_path)
