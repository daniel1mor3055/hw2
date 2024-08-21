import torch
import numpy as np


class FastDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alphas, self.alphas_cumprod, self.sigmas = self.calc_diffusion_hyperparams()
        self.timesteps = np.arange(0, self.num_train_timesteps)[::-1]
        self.init_noise_sigma = 1.0  # Standard deviation of the initial noise
        self.order = 1  # FastDPM is a first-order method

    def set_timesteps(self, num_inference_steps, device=None):
        # Set the device if provided
        if device:
            self.device = device

        # Set the inference steps (like steps_list in your original code)
        self.num_inference_steps = num_inference_steps
        self.timesteps = self.get_STEP_step(self.num_inference_steps, self.num_train_timesteps)

    def get_schedule(self, timesteps):
        alphas_cumprod = self.alphas_cumprod
        return alphas_cumprod[timesteps], torch.sqrt(1 - alphas_cumprod[timesteps])

    def step(self, model_output, timestep, sample):
        next_timestep = timestep - 1 if timestep > 0 else 0
        alpha = self.alphas_cumprod[timestep]
        alpha_next = self.alphas_cumprod[next_timestep]

        beta = 1 - alpha
        sigma = self.sigmas[timestep]

        pred_original_sample = (sample - torch.sqrt(1 - alpha) * model_output) / torch.sqrt(alpha)
        prev_sample = (
                torch.sqrt(alpha_next) * pred_original_sample +
                torch.sqrt(1 - alpha_next) * torch.randn_like(sample)
        )

        return prev_sample

    def calc_diffusion_hyperparams(self):
        beta = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, device=self.device)
        alpha = 1.0 - beta
        alphas_cumprod = torch.cumprod(alpha, dim=0)
        sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        return alpha, alphas_cumprod, sigmas

    def get_STEP_step(self, S, T):
        c = (T - 1.0) / (S - 1.0)
        list_tau = [np.floor(i * c) for i in range(S)]
        return torch.tensor(list_tau, dtype=torch.long, device=self.device)

    def scale_model_input(self, sample, timestep):
        return sample
