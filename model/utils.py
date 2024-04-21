import torch

def collate_fn(examples):
    """
    Collate Function is used to create a batch
    """
    pixel_values = torch.stack([torch.Tensor(example["image_tensor"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([torch.Tensor(example["tokens"]).to(torch.int64) for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}