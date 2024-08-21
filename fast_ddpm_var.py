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


def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value

    Returns:
    x (float)
    """
    #
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(
    S,
    diffusion_config: DiffusionParams,
):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic

    Returns:
    np array of noise levels, size = (S, )
    """
    beta_0, beta_T = diffusion_config.beta_start, diffusion_config.beta_end
    T = diffusion_config.num_timesteps
    target = np.prod(1 - np.linspace(beta_0, beta_T, T))

    g = lambda x: np.linspace(beta_0, x, S)
    domain = (beta_0, 0.99)

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


# VAR
def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32).to(device)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t - 1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]

    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user - 1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i + 1]:
                    t_adapted = bisearch(
                        f=lambda _t: _log_cont_noise(
                            _t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T
                        ),
                        domain=(i - 0.01, i + 1.01),
                        target=np.log(Gamma_bar[t].cpu().numpy()),
                    )
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps


def VAR_sampling(net, size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):
    """
    Perform the complete sampling step according to user defined variances

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_eta (np.array):    User defined noise
    kappa (float):                  factor multipled over sigma, between 0 and 1
    continuous_steps (list):        continuous steps computed from user_defined_eta

    Returns:
    the generated images in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32).to(device)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t - 1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]

    # print('begin sampling, total number of reverse steps = %s' % T_user)

    x = std_normal(size)
    with torch.no_grad():
        for i, tau in enumerate(continuous_steps):
            diffusion_steps = tau * torch.ones(size[0]).to(device)
            epsilon_theta = net(x, diffusion_steps)
            if i == T_user - 1:  # the next step is to generate x_0
                assert abs(tau) < 0.1
                alpha_next = torch.tensor(1.0)
                sigma = torch.tensor(0.0)
            else:
                alpha_next = Gamma_bar[T_user - 1 - i - 1]
                sigma = kappa * torch.sqrt(
                    (1 - alpha_next)
                    / (1 - Gamma_bar[T_user - 1 - i])
                    * (1 - Gamma_bar[T_user - 1 - i] / alpha_next)
                )
            x *= torch.sqrt(alpha_next / Gamma_bar[T_user - 1 - i])
            c = torch.sqrt(1 - alpha_next - sigma**2) - torch.sqrt(
                1 - Gamma_bar[T_user - 1 - i]
            ) * torch.sqrt(alpha_next / Gamma_bar[T_user - 1 - i])
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
    user_defined_eta = get_VAR_noise(S=S, diffusion_config=diffusion_config)
    for i in tqdm(range(n_generate // batchsize), desc=f"{S}_sampling_timestamps"):
        continuous_steps = _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta)
        Xi = VAR_sampling(
            model,
            (batchsize, C, H, W),
            diffusion_hyperparams,
            user_defined_eta,
            kappa=1,
            continuous_steps=continuous_steps,
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
            f"fast_dpm_var_sampling_{S}",
        )

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with torch.no_grad():
            generate(config.diffusion_params, S, n_generate, sampling_batch_size, save_dir, model)
