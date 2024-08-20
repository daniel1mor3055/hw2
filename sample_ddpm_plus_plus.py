import os

import torch
import torchvision
from diffusers import DPMSolverMultistepScheduler
from tqdm import tqdm

from consts import config
from unet_base import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(model, config, save_dir, num_timesteps):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save each image individually as x0 predictions in batches.
    """

    model_config = config.model_params
    sampling_config = config.sampling_params

    assert (
        sampling_config.num_samples % sampling_config.sampling_batch_size == 0
    ), "num_samples must ba a multiple of sampling_batch_size"

    num_batches = sampling_config.num_samples // sampling_config.sampling_batch_size

    for batch_idx in tqdm(range(num_batches)):
        # Create the noise scheduler
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_params.num_timesteps
        )
        scheduler.config.algorithm_type = "dpmsolver++"
        scheduler.set_timesteps(num_timesteps)

        xt = torch.randn(
            (
                sampling_config.sampling_batch_size,
                model_config.im_channels,
                model_config.im_size,
                model_config.im_size,
            )
        ).to(device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                # Get prediction of noise
                noisy_image = model(xt, t.unsqueeze(0).to(device))

                # Use scheduler to get x_prev
                prev_noisy_sample = scheduler.step(noisy_image, t, xt).prev_sample
                xt = prev_noisy_sample

        ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
        ims = (ims + 1) / 2

        for j in range(ims.size(0)):
            img = torchvision.transforms.ToPILImage()(ims[j])
            img.save(
                os.path.join(
                    save_dir, f"sample_{batch_idx * sampling_config.sampling_batch_size + j}.png"
                )
            )
            img.close()


def infer():
    model_config = config.model_params
    train_config = config.train_params

    model = Unet(model_config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(train_config.task_name, train_config.ckpt_name), map_location=device
        )
    )
    model.eval()

    # Iterate through different num_timesteps configurations
    for num_timesteps in config.diffusion_params.num_timesteps_list:
        # Set directory name based on num_timesteps
        save_dir = os.path.join(
            train_config.task_name,
            f"dpm_pp_sampling_{num_timesteps}",
        )

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with torch.no_grad():
            sample(model, config, save_dir, num_timesteps)


if __name__ == "__main__":
    infer()
