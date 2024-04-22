"""
This module has the custom Stable Diffusions Training Configuration and pipeline
To understand the architecture of Stable Diffusions and the training process, we
used the reference
- https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work
"""
from typing import Any, Dict, Tuple
import os
from datasets import load_dataset

# diffusers model
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPipeline
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPFeatureExtractor
)
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm.auto import tqdm
from .utils import collate_fn

class CustomStableDiffusionTraining:
    def __init__(self, configs: Dict):
        """
        instantiate this instance of CustomStableDiffusionTraining with
        - an Accelerator
        - components of Stable Diffusion model:
            - a CLIPTokenizer: for translating caption text to the embedding space
            - a VAE: map the image to a Latent space using the model's encoder
            - a CLIPTextModel: for generating caption's embedding hidden state, which is
            used to conditioned the training process
            - a UNET: is the model we need to finetune (thus set to train mode).
            - a noise scheduler: for denoising the noisy latent produced by UNET
            back to the images
        - an optimizer
        - a learning-rate scheduler
        :param configs: is a Dictionary parsed from `./configs/experiment.yaml` file
        """
        self.accelerator = Accelerator(gradient_accumulation_steps=4,
                                        log_with=LoggerType.TENSORBOARD,
                                        project_dir=configs["output"]["log"])

        print("==========================")
        print("Process ID:", os.getpid())
        print("Parent Process ID:", os.getppid())
        print("==========================")
        scaled_lr = configs["training"]["learning_rate"] * self.accelerator.num_processes
        # init tracker
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("genai-for-theme",
            config={"num_iterations": configs["training"]["max_train_steps"],
            "learning_rate": scaled_lr})
 
        self.configs: Dict = configs
        self.weight_dtype = torch.float32
        self.train_loader = self._init_train_loader()
        self.unet = UNet2DConditionModel.from_pretrained(
            **configs["model"]["unet"]
        )
        self.vae = AutoencoderKL.from_pretrained(
            **configs["model"]["vae"]
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            **configs["model"]["clip"]
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            **configs["model"]["tokenizer"]
        )
        self.noise_scheduler = self._get_noise_scheduler(
            configs["model"]["noise_scheduler"]["name"],
            **configs["model"]["noise_scheduler"]["args"]
        )
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=scaled_lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
        self.progress_bar = tqdm(range(self.configs["training"]["max_train_steps"]),
                                disable=not self.accelerator.is_local_main_process)
        self.progress_bar.set_description("Steps")
        self.global_step = 0

    def _init_train_loader(self):
        """
        Call the Dataset.load_dataset api to download `.parquet` train dataset
        from a path, configured in `./configs/experiment.yaml` file.
        Then, instantiate the train_loader to be used in the `_train_each_epoch()` loop
        """
        train_set = load_dataset(self.configs["data"]["type"],\
                                 data_dir=self.configs["data"]["path"])
        train_set = train_set["train"]
        return torch.utils.data.DataLoader(train_set,
                                           batch_size=self.configs["training"]["batch_size"],
                                           shuffle=True, collate_fn=collate_fn, num_workers=2)

    def _get_noise_scheduler(self, type: str, **kwargs):
        """
        instantiate the the Noise Scheduler depending on the noise_scheduler.name
        in `./configs/experiment.yaml`
        """
        if type == "PNDMScheduler":
            return PNDMScheduler(**kwargs)
        if type == "DDIMScheduler":
            return DDIMScheduler(**kwargs)
        if type == "LMSDiscreteScheduler":
            return LMSDiscreteScheduler(**kwargs)

    def _create_pipeline(self, unet):
        """
        Once finetune is completed, we can call this method to create an upload
        the pipeline to a shared drive or to the hub. The ouput location if configured
        in ./configs/experiment.yaml output.model
        :param unet: is the Accelerate's prepared instance of self.unet
        """
        version = self.configs["version"]
        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                text_encoder= self.accelerator.unwrap_model(self.text_encoder),
                vae=self.vae,
                unet=unet.module if self.accelerator.num_processes >1 else self.unet,
                tokenizer=self.tokenizer,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
            pipeline.save_pretrained(os.path.join(self.configs['output']['model'],
                                                    f"stable-diffusion-{version}"))

    def _train_each_epoch(self, unet,
                                optimizer,
                                train_loader,
                                lr_scheduler) -> Tuple:
        """
        This method has operations that happen each training epoch
        :param unet: is the Accelerate's prepared instance of self.unet
        :param optimizer: is the Accelerate's prepared instance of self.optimizer
        :param train_loader: is the Accelerate's prepared instance of self.train_loader
        :param lr_scheduler: is the Accelerate's prepared instance of self.lr_scheduler
        :return tuple: return back a tuple of (unet, optimizer, train_loader, lr_scheduler) to the caller
        This training step is referenced and adapted from:
        - https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/4_stable_diffusion/src/finetune.py
        - https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

        """
        for step, batch in enumerate(train_loader):
            with self.accelerator.accumulate(unet):
                # First encode the image to laten space with the VAE encoder
                latent = self.vae.encode(batch["pixel_values"].to(self.weight_dtype))\
                                .latent_dist.sample().detach()

                latent = latent * 0.18215
                # sample noise to add to latent
                noise = torch.randn(latent.shape).to(latent.device)
                bsz = latent.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0,
                                            self.noise_scheduler.config.num_train_timesteps,
                                            (bsz,),
                                            device=latent.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latent, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                # Predict the noise residual and compute loss
                model_pred = self.unet(noisy_latents,
                                    timesteps,
                                    encoder_hidden_states,
                                    return_dict=True)["sample"]


                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                loss = loss.mean([1, 2, 3]).mean()

                # Backpropagate
                self.accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                for _ in range(self.accelerator.num_processes):
                    self.progress_bar.update(1)
                    self.global_step += 1
    
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)
                self.accelerator.clip_grad_norm_(unet.parameters(), 1)
                self.accelerator.log(logs, step=self.global_step)

            # TODO: Early termination
            # if exceed max_step size or loss hasnt been improving
        # if self.accelerator.is_main_process:
        #     with open(self.configs["output"]["log"]) as ofile:
        #         ofile.write(f"{logs}")

        self.accelerator.wait_for_everyone()
        return (unet,
                optimizer,
                train_loader,
                lr_scheduler)

    def train(self) -> None:
        """
        This function start the training process for Stable Diffusion model.
        The main operations carried out in the training process are:
        - Freeze parameters for Variational Auto Encoder model (vae)
        - Freeze parameters for CLIPText encoding model (text_encoder)
        - Setup the model to be run on `device` (either CUDA or CPU)
        - Wrap the Unet model, train_loader, and lr_scheduler with accelerate
        to prepare for distributed training
        - Create a StableDiffusionPipeline object and save to the `self.configs.output.model`
        path, as configured in the `experiment.yaml` file
        """
        # freeze params
        for params in self.vae.parameters():
            params.requires_grad = False

        for params in self.text_encoder.parameters():
            params.requires_grad = False

        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)

        unet, optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_loader, self.lr_scheduler
        )

        self.accelerator.wait_for_everyone()
        for epoch in range(self.configs["training"]["epochs"]):
            print("EPOCH", epoch)
            unet, optimizer, train_loader, lr_scheduler = self._train_each_epoch(unet,
                                                                                optimizer,
                                                                                train_loader,
                                                                                lr_scheduler)
            if self.global_step >= self.configs["training"]["max_train_steps"]:
                break
        self._create_pipeline(unet)
        self.accelerator.end_training()
