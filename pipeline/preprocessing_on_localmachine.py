from datasets.arrow_dataset import Dataset
import torchvision.transforms as T
from datasets import load_dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from configs.cs777_genai_paths import DATA_FOLDER, TRAIN_SET_FOLDER
import torch
import pyarrow.parquet as pq
import os
from transformers import CLIPTokenizer

# Load dataset
img_dataset: Dataset = load_dataset("imagefolder", data_dir=DATA_FOLDER, split="train")

# Resize and pad function
def resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0):
    original_width, original_height = img.size
    max_side = max(original_width, original_height)

    if original_width > original_height:
        scale = max_side / original_width
        new_width = max_side
        new_height = int(original_height * scale)
    else:
        scale = max_side / original_height
        new_height = max_side
        new_width = int(original_width * scale)

    img = F.resize(img, (new_height, new_width), interpolation=InterpolationMode.LANCZOS)

    pad_width = (max_side - new_width) // 2
    pad_height = (max_side - new_height) // 2

    img = F.pad(img, [pad_width, pad_height, pad_width, pad_height], padding_mode=padding_mode, fill=fill)
    img = F.resize(img, final_size, interpolation=InterpolationMode.LANCZOS)
    return img

# Transformation pipeline
transform_pipeline = T.Compose([
    T.Lambda(lambda img: resize_and_pad_then_resize(img, final_size=(512, 512), padding_mode='constant', fill=0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Preprocessing function
def preprocess_train(examples):
    examples["pixel_values"] = [transform_pipeline(image) for image in examples["image"]]
    inputs = tokenizer(examples["caption"], padding="max_length", truncation=True, return_tensors="pt")
    examples["input_ids"] = inputs.input_ids
    return examples

# Map and preprocess
train_set = img_dataset.map(preprocess_train, batched=True, remove_columns=["image", "caption"])

# Save as parquet
if not os.path.exists(TRAIN_SET_FOLDER):
    os.makedirs(TRAIN_SET_FOLDER)
pq.write_table(train_set.data.table, 
               os.path.join(TRAIN_SET_FOLDER, 'new_train_set.parquet'))

# # Read parquet
# train_set_read_from_parquet = Dataset(pq.read_table(os.path.join(TRAIN_SET_FOLDER, 'new_train_set.parquet')))

# # Check the shape of pixel_values
# print(type(train_set_read_from_parquet[0]['pixel_values']))
# print(train_set_read_from_parquet[0]['pixel_values'])

# # Convert pixel_values to tensor and visualize
# new_img = train_set_read_from_parquet[0]['pixel_values']

# img_tensor = torch.tensor(new_img)
# tensor = img_tensor.squeeze(0)
# unloader = T.ToPILImage()
# image = unloader(tensor)
# image
