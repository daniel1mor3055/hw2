import argparse
import operator
import os

import torch
import torchvision
import yaml
from tqdm import tqdm
from consts import num_timesteps_list

from linear_noise_scheduler import LinearNoiseScheduler
from unet_base import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(model, scheduler, config, save_dir):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save each image individually as x0 predictions in batches.
    """

    train_config, model_config, diffusion_config, sampling_config = operator.itemgetter(
        "train_params", "model_params", "diffusion_params", "sampling_params"
    )(config)

    assert (
        sampling_config["num_samples"] % sampling_config["sampling_batch_size"] == 0
    ), "num_samples must ba a multiple of sampling_batch_size"
    batch_size = sampling_config["sampling_batch_size"]
    num_batches = sampling_config["num_samples"] // batch_size

    for batch_idx in tqdm(range(num_batches)):
        xt = torch.randn(
            (
                batch_size,
                model_config["im_channels"],
                model_config["im_size"],
                model_config["im_size"],
            )
        ).to(device)

        for i in reversed(range(diffusion_config["num_timesteps"])):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            # Use scheduler to get x0 and xt-1
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save each image individually
        ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
        ims = (ims + 1) / 2

        for j in range(ims.size(0)):
            img = torchvision.transforms.ToPILImage()(ims[j])
            img.save(os.path.join(save_dir, f"sample_{batch_idx * batch_size + j}.png"))
            img.close()


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
        config["diffusion_params"]["num_timesteps"] = num_timesteps

        # Create the noise scheduler
        scheduler = LinearNoiseScheduler(**config["diffusion_params"])

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
            sample(model, scheduler, config, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument("--config", dest="config_path", default="config.yaml", type=str)
    args = parser.parse_args()
    infer(args)
