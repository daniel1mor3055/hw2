import os

import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
)

from FastDPMScheduler import FastDPMScheduler

# Define the prompt and the different number of steps to test
prompt = "A cat wearing a spacesuit"
steps_list = [5, 10, 50, 100]
output_dir = "./stable_diffusion_outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the pretrained Stable Diffusion 2.1 model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# Dictionary of samplers to test
samplers = {
    # "vanilla": DDPMScheduler.from_config(pipe.scheduler.config),
    # "dpmsolver++": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    # "ddim": DDIMScheduler.from_config(pipe.scheduler.config),
    "fastdpm": FastDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
}

# Test each sampler
for sampler_name, scheduler in samplers.items():
    if scheduler is None:
        print(f"Skipping {sampler_name} as it is not available in diffusers.")
        continue

    pipe.scheduler = scheduler
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    for steps in steps_list:
        print(f"Generating images with {sampler_name} for {steps} steps...")

        # Update the scheduler with the desired number of steps
        pipe.scheduler.set_timesteps(steps)

        # Generate image
        with torch.no_grad():
            image = pipe(prompt=prompt, num_inference_steps=steps).images[0]

        # Save the image
        image_save_path = os.path.join(output_dir, f"{sampler_name}_steps_{steps}.png")
        image.save(image_save_path)
        print(f"Saved image at: {image_save_path}")

print("Done with all samplers!")
