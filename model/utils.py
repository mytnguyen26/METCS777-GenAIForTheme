import torch

def collate_fn(examples):
    """
    Collate Function is used to create a batch
    """
    pixel_values = torch.stack([torch.Tensor(example["pixel_values"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([torch.Tensor(example["input_ids"]).to(torch.int64) for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}