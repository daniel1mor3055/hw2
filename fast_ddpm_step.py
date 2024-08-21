import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from consts import DiffusionParams, config
from unet_base import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rescale(X, batch=True):
    if not batch:
        return (X - X.min()) / (X.max() - X.min())
    else:
        for i in range(X.shape[0]):
            X[i] = rescale(X[i], batch=False)
        return X


def std_normal(size):
    return torch.normal(0, 1, size=size).to(device)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
        T,
        Beta,
        Alpha,
        Alpha_bar,
        Sigma,
    )
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def get_STEP_step(S, T):
    c = (T - 1.0) / (S - 1.0)
    list_tau = [np.floor(i * c) for i in range(S)]

    return [int(s) for s in list_tau]


def STEP_sampling(net, size, diffusion_hyperparams, user_defined_steps, kappa):
    """
    Perform the complete sampling step according to https://arxiv.org/pdf/2010.02502.pdf
    official repo: https://github.com/ermongroup/ddim

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_steps (int list):  User defined steps (sorted)
    kappa (float):                  factor multipled over sigma, between 0 and 1

    Returns:
    the generated images in torch.tensor, shape=size
    """
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, _ = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    T_user = len(user_defined_steps)
    user_defined_steps = sorted(list(user_defined_steps), reverse=True)

    x = std_normal(size)
    with torch.no_grad():
        for i, tau in enumerate(user_defined_steps):
            diffusion_steps = tau * torch.ones(size[0]).to(device)
            epsilon_theta = net(x, diffusion_steps)
            if i == T_user - 1:  # the next step is to generate x_0
                assert tau == 0
                alpha_next = torch.tensor(1.0)
                sigma = torch.tensor(0.0)
            else:
                alpha_next = Alpha_bar[user_defined_steps[i + 1]]
                sigma = kappa * torch.sqrt(
                    (1 - alpha_next) / (1 - Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next)
                )
            x *= torch.sqrt(alpha_next / Alpha_bar[tau])
            c = torch.sqrt(1 - alpha_next - sigma**2) - torch.sqrt(
                1 - Alpha_bar[tau]
            ) * torch.sqrt(alpha_next / Alpha_bar[tau])
            x += c * epsilon_theta + sigma * std_normal(size)
    return x


def generate(diffusion_config: DiffusionParams, S, n_generate, batchsize, save_dir, model):

    assert n_generate % batchsize == 0, "num_samples must be a multiple of sampling_batch_size"
    model_config = config.model_params

    diffusion_hyperparams = calc_diffusion_hyperparams(
        T=diffusion_config.num_timesteps,
        beta_0=diffusion_config.beta_start,
        beta_T=diffusion_config.beta_end,
    )
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    C, H, W = model_config.im_channels, model_config.im_size, model_config.im_size
    user_defined_steps = get_STEP_step(S=S, T=diffusion_config.num_timesteps)
    for i in tqdm(range(n_generate // batchsize), desc=f"{S}_sampling_timestamps"):
        Xi = STEP_sampling(
            model,
            (batchsize, C, H, W),
            diffusion_hyperparams,
            user_defined_steps,
            kappa=0,
        )
        for j, x in enumerate(rescale(Xi)):
            index = i * batchsize + j
            img = torchvision.transforms.ToPILImage()(x)
            img.save(os.path.join(save_dir, f"sample_{index}.png"))
            img.close()


if __name__ == "__main__":

    model_config = config.model_params
    train_config = config.train_params

    model = Unet(model_config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(train_config.task_name, train_config.ckpt_name), map_location=device
        )
    )
    model.eval()
    for S in config.diffusion_params.num_timesteps_list:
        n_generate = config.sampling_params.num_samples
        sampling_batch_size = config.sampling_params.sampling_batch_size

        # Set directory name based on num_timesteps
        save_dir = os.path.join(
            train_config.task_name,
            f"fast_dpm_sampling_{S}",
        )

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with torch.no_grad():
            generate(config.diffusion_params, S, n_generate, sampling_batch_size, save_dir, model)
