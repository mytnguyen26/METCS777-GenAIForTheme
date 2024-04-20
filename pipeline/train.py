# to load and transform image datasets to tensor
from datasets import load_dataset, load_from_disk, VerificationMode
from datasets.arrow_dataset import Dataset
# diffusers model
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# Text processing
from transformers import CLIPTextModel, CLIPTokenizer

# Image processing
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import ConstantLR
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import LMSDiscreteScheduler

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from configs import DATA_FOLDER
from preprocess import *

def train():

    # Getting the model weights from the pretrained models hub
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    unet.enable_gradient_checkpointing()

    # Freeze vae and text_encoder (we only train the UNET)

    for params in vae.parameters():
        params.requires_grad = False

    for params in text_encoder.parameters():
        params.requires_grad = False
        
    noise_scheduler = PNDMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    skip_prk_steps=True)


    lr = 0.001
    batch_size = 4
    global_step = 0
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    lr_scheduler = ConstantLR(optimizer)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate_fn, num_workers=1)
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    epochs = 2
    weight_dtype = torch.float32

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    prediction_type = "v_prediction"

    # Move vae and unet to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    accelerator.wait_for_everyone()
    for epoch in range(epochs):
        print("EPOCH", epoch)
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                # First encode the image to laten space with the VAE encoder
                latent = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()

                # sample noise to add to latent
                noise = torch.randn_like(latent)
                bsz = latent.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latent.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=prediction_type)

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                for _ in range(accelerator.num_processes):
                    # progress_bar.update(1)
                    global_step += 1 

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                # progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                accelerator.clip_grad_norm_(unet.parameters(), 1)

            # if global_step >= args.max_train_steps:
            #     break
        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet.module if accelerator.num_processes >1 else unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            # safety_checker=safety_checker,
            # feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(OUTPUT_DIR)

    accelerator.end_training()

if __name__=="__main__":
    img_dataset: Dataset = load_dataset("imagefolder", data_dir=DATA_FOLDER, split="train")
    
    train_set = img_dataset.with_transform(preprocess_train)

    train()