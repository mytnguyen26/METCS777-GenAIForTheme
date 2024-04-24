import torch
from diffusers import StableDiffusionPipeline


if __name__=="__main__":
    inference = StableDiffusionPipeline.from_pretrained(
        "./output/stable-diffusion-4", torch_dtype=torch.float32
    ).to("cuda")

    img = inference("a horse running on big open field", num_inference_steps=100).images[0]
    
    img.save("./output/test_2.png")