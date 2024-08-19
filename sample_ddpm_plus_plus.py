import argparse
import os

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from consts import num_timesteps_list

from diffusers import DPMSolverMultistepScheduler
from unet_base import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(model, scheduler):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save each image individually as x0 predictions in batches.
    """
    scheduler.set_timesteps(50)

    noise = torch.randn((1, 1, 28, 28), device=device)
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_image = model(input, t.unsqueeze(0).to(device))
            prev_noisy_sample = scheduler.step(noisy_image, t, input).prev_sample
            input = prev_noisy_sample

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    print("stop here")


def infer(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    model_config = config["model_params"]
    train_config = config["train_params"]
    sampling_config = config["sampling_params"]

    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(train_config["task_name"], train_config["ckpt_name"]), map_location=device
        )
    )
    model.eval()

    # Iterate through different num_timesteps configurations

    for num_timesteps in num_timesteps_list:
        # Update diffusion_config with current num_timesteps
        diffusion_config = {
            "num_timesteps": num_timesteps,
            "beta_start": config["diffusion_params"]["beta_start"],
            "beta_end": config["diffusion_params"]["beta_end"],
        }

        # Create the noise scheduler
        scheduler = DPMSolverMultistepScheduler()
        scheduler.config.algorithm_type = "dpmsolver++"

        # Set directory name based on num_timesteps
        save_dir = os.path.join(
            train_config["task_name"],
            f"{sampling_config['sampling_algorithm']}_sampling_{num_timesteps}",
        )

        # Create output directories
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Run sampling
        with torch.no_grad():
            sample(model, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument("--config", dest="config_path", default="config.yaml", type=str)
    args = parser.parse_args()
    infer(args)
