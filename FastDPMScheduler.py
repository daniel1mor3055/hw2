import numpy as np
import torch


class FastDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None):
        self.num_train_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_hyperparams = self.calc_diffusion_hyperparams()
        self.init_noise_sigma = 1.0
        self.order = 1

    def set_timesteps(self, S, device=None):
        # Set the device if provided
        if device:
            self.device = device

        # Set the inference steps (like steps_list in your original code)
        self.S = S
        self.timesteps = self.get_STEP_step(self.S, self.num_train_timesteps)

    def std_normal(self, size):
        return torch.normal(0, 1, size=size).to(self.device)

    def get_schedule(self, timesteps):
        alphas_cumprod = self.alphas_cumprod
        return alphas_cumprod[timesteps], torch.sqrt(1 - alphas_cumprod[timesteps])

    def step(self, model_output, timestep, sample, return_dict=False, kappa=0.0):
        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        if timestep == self.timesteps[-1]:  # the next step is to generate x_0
            alpha_next = torch.tensor(1.0)
            sigma = torch.tensor(0.0)
        else:
            alpha_next = Alpha_bar[self.timesteps[(self.timesteps == timestep).nonzero(as_tuple=True)[0].item() + 1]]
            sigma = kappa * torch.sqrt(
                (1 - alpha_next) / (1 - Alpha_bar[timestep.item()]) * (1 - Alpha_bar[timestep.item()] / alpha_next)
            )
        sample *= torch.sqrt(alpha_next / Alpha_bar[timestep.item()])
        c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(
            1 - Alpha_bar[timestep.item()]
        ) * torch.sqrt(alpha_next / Alpha_bar[timestep.item()])
        sample += c * model_output + sigma * self.std_normal(sample.size())

        return sample

    def calc_diffusion_hyperparams(self):
        Beta = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, self.num_train_timesteps):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        _dh = {}
        _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
            self.num_train_timesteps,
            Beta,
            Alpha,
            Alpha_bar,
            Sigma,
        )
        diffusion_hyperparams = _dh
        return diffusion_hyperparams

    def get_STEP_step(self, S, T):
        c = (T - 1.0) / (S - 1.0)
        list_tau = [np.floor(i * c) for i in range(S)][::-1]
        return torch.tensor(list_tau, dtype=torch.long, device=self.device)

    def scale_model_input(self, sample, _timestep):
        return sample
