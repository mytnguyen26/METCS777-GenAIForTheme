# Text processing
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

transform_pipeline = v2.Compose([
        # TODO
        # Instead of resize, enlarge the photo by ratio and add padding
        v2.Resize(size=(512, 512), interpolation=InterpolationMode.LANCZOS),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    examples["input_ids"] = inputs.input_ids
    return examples

def collate_fn(examples):
    """
    Collate Function is used to create a batch
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}