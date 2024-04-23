from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
import boto3
from datasets import load_dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from transformers import CLIPTokenizerFast
from torchvision.transforms import InterpolationMode
import pandas as pd
from datasets import Dataset
import pyarrow.parquet as pq

# Initialize Spark context
sc = SparkContext(appName="PySpark Image Processing with RDDs")
# Initialize Spark session
spark = SparkSession.builder \
    .appName("PySpark Image Processing with RDDs and DataFrames") \
    .getOrCreate()

# Define bucket and paths
bucket_name = 'caozhen'
raw_data_path = 'raw_data/harvard/paintings/'
train_set_save_path = 'train_set/train_set.parquet'

# Define local folder
local_folder = 'paintings'
os.makedirs(local_folder, exist_ok=True)

# Define S3 client
s3_client = boto3.client('s3')


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


# Download images from S3
download_s3_folder(bucket_name, raw_data_path, local_folder)

# Load dataset using the datasets library
img_dataset = load_dataset("imagefolder", data_dir=local_folder, split="train")

# Create an RDD from the dataset
rdd = sc.parallelize(list(zip(img_dataset['image'], img_dataset['caption'])))


def resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0):
    """
    Resize an image to make its longest side equal to the original image's longest side,
    pad the shorter side to make the image a square, then resize to final_size.

    Args:
        img (PIL.Image): The image to resize and pad.
        final_size (tuple): The desired output size (height, width).
        padding_mode (str): Type of padding. Options include 'constant', 'edge', etc.
        fill (int, tuple): Pixel fill value for constant padding. Can be int or tuple.

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
    img = F.resize(img, (new_height, new_width), interpolation=InterpolationMode.LANCZOS)

    # Calculate padding amounts
    pad_width = (max_side - new_width) // 2
    pad_height = (max_side - new_height) // 2

    # Apply padding to make it a square
    img = F.pad(img, [pad_width, pad_height, pad_width, pad_height], padding_mode=padding_mode, fill=fill)

    # Final resize to the desired output size
    img = F.resize(img, final_size, interpolation=InterpolationMode.LANCZOS)
    return img


def preprocess_image(image_path, caption):
    # Setup the transformation pipeline with the updated function
    transform_pipeline = T.Compose([
        T.Lambda(lambda img: resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = transform_pipeline(image_path)

    # Tokenization
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer.encode(caption, max_length=77, truncation=True, return_tensors="pt").tolist()[0]

    return (processed_image, tokens)


# Map transformation function over RDD
processed_rdd = rdd.map(lambda x: preprocess_image(x[0], x[1]))
dataset = processed_rdd.collect()

formatted_data = {
    'pixel_values': [item[0].tolist() for item in dataset],
    'input_ids': [item[1] for item in dataset]
}

df = pd.DataFrame(formatted_data)
train_set = Dataset.from_pandas(df)

# Upload to S3 parquet
pq.write_table(train_set.data.table, 'train_set.parquet')
s3_client.upload_file('train_set.parquet', bucket_name, train_set_save_path)

# Stop the Spark context
sc.stop()
